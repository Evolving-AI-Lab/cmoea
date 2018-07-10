/*
 * modules/cmoea/mp_cmoea_nsga2_mpi.hpp
 *
 *  Created on: Mar 10, 2017
 *      Author: Joost Huizinga
 *
 * This is the file for the massively parallel version of C-MOEA.
 * The main idea behind MP C-MOEA, is that there is no longer a master and a set of workers.
 * Instead, each worker will create its own population, and occasionally exchange part of the,
 * population with other workers.
 */

#ifndef MODULES_CMOEA_CMOEA_NSGA2_MPI_HPP_
#define MODULES_CMOEA_CMOEA_NSGA2_MPI_HPP_

// Standard includes
#include <algorithm>
#include <limits>

// Boost includes
#include <boost/foreach.hpp>
#include <boost/multi_array.hpp>
#include <boost/array.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/spirit/include/karma.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/detail/point_to_point.hpp>

// Sferes includes
#include <sferes/stc.hpp>
#include <sferes/ea/ea.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/dbg/dbg.hpp>
#include <sferes/ea/dom_sort_basic.hpp>
#include <sferes/ea/common.hpp>
#include <sferes/ea/crowd.hpp>

// Module includes
#include <modules/misc/common_compare.hpp>
#include <modules/debugext/dbgext.hpp>

// Local includes
#include "mpi_util.hpp"
#include "nsga_util.hpp"

namespace karma = boost::spirit::karma;

// Debug statements become a bit too long without this statement
using namespace dbg;
using namespace dbgext;
#define DBO dbo("ea") << "["<< _world->rank() << "] "
#define DBOF dbo("eafit") << "["<< _world->rank() << "] "
#define DBOO dbo("eaobj") << "["<< _world->rank() << "] "
#define DBO_MPI dbo("mpi") << "["<< _world->rank() << "] "

namespace sferes
{
namespace ea
{


/**
 * This is the main class for massively parallel C-MOEA.
 *
 * Usually, this part is only executed by the master.
 */
SFERES_EA(MPCmoeaNsga2Mpi, Ea){
public:
    /** PARAMETERS **/

    // The type of Pareto domination sort to use.
    // Currently available types are:
    // - sferes::ea::dom_sort_basic_f
	//   (defined in sferes/ea/dom_sort_basic.hpp)
    //   Sorts according to pareto dominance and will add individuals from the
	//   highest to the lowest layer, with crowding as a tie-breaker in the last
	//   layer to be added.
    // - sferes::ea::dom_sort_no_duplicates_f
	//   (defined in sferes/ea/dom_sort_no_duplicates.hpp)
    //   Same as dom_sort_basic, accept that, for each front, only one
	//   individual per pareto location is added. Other individuals at the same
	//   location will be bumped to the next layer.
    typedef typename Params::ea::dom_sort_f dom_sort_f;

    // The type non dominated comparator to use
    // Currently available types are:
    // - sferes::ea::_dom_sort_basic::non_dominated_f
    //   (defined in sferes/ea/dom_sort_basic.hpp)
    //   Regular comparisons based on dominance
    // - sferes::ea::cmoea_nsga::prob_dom_f<Params>
    //   Comparisons based on probabilistic sorting, where some objectives can
    //   be stronger than others.
    typedef typename Params::cmoea_nsga::non_dom_f non_dom_f;

    // The index used to temporarily store the category
    // This index should hold a dummy value, as it will be overwritten
    // constantly. The default would be 0.
    static const size_t obj_index = Params::cmoea::obj_index;

    // The number of objectives (bins) used in the map
    size_t nr_of_bins;

    // The size of each bin
    static const size_t bin_size = Params::cmoea::bin_size;

    // The number of individuals initially generated to fill the archive
    // If equal to the bin_size, every initially generated individual is added
    // to every bin of every category.
    // The init_size has to be greater than or equal to the bin_size
    static const size_t init_size = Params::pop::init_size;

    // Very large initial populations may cause CMOEA to run out of memory.
    // To avoid this, you can add the initial populations in init_batch batches
    // of init_size.
    //static const size_t init_batch = Params::pop::init_batch;

    // Max batch size to avoid memory issues
    static const size_t max_batch_size = Params::pop::max_batch_size;

    // Individuals generated each epoch per bin (needs to be divisible by 2)
    static const size_t offspring_per_generation = Params::pop::select_size;

    // The number of replicates for each bin
    static const int replicates = Params::cmoea::replicates;

    // Due to memory limitations, you may not always be able to send the entire
    // population at once. max_send_size determines the maximum size of the
    // population to send at once.
    static const size_t max_send_size = Params::cmoea::max_send_size;

    // The total number of bins, including replicates
    int bins_total;


    /** TYPE DEFINITIONS **/

    // The type of the phenotype
    typedef Phen phen_t;

    // A smart pointer to the phenotype
    typedef boost::shared_ptr<Phen> phen_ptr_t;
    typedef phen_ptr_t indiv_t;

    // The population type
    typedef typename std::vector<phen_ptr_t> pop_t;

    // Iterator for the population
    typedef typename std::vector<phen_ptr_t>::iterator pop_it_t;

    // The type of the Pareto front
    typedef typename std::vector<pop_t> front_t;

    // The type of a collections of bins
    typedef typename std::vector<pop_t> bins_t;

    // The type of the mpi environment
    typedef typename boost::mpi::environment mpi_env_t;

    // The type of the mpi communicator
    typedef typename boost::mpi::communicator mpi_com_t;

    // The type of serialization archive for sending
    typedef typename boost::mpi::packed_oarchive mpi_oarchive_t;

    // The type of serialization archive for receiving
    typedef typename boost::mpi::packed_iarchive mpi_iarchive_t;

    // The exact type of non-dominated sort function to use
    typedef fill_dom_sort_f<phen_ptr_t, dom_sort_f, non_dom_f> sort_f;

    // Maps MPI instances to bin indices
    typedef std::map<int, std::vector<int> > inst_bin_map_t;

    // Friend declaration required for Sferes
    SFERES_EA_FRIEND(MPCmoeaNsga2Mpi);

    /**
     * Constructor for MP-CMOEA.
     *
     * Initializes the MPI environment, and calculates some relevant statistics
     * for this run.
     */
    MPCmoeaNsga2Mpi(){
        dbg::trace trace("ea", DBG_HERE);

        // Initialize MPI
		static char* argv[] = {(char*)"sferes2", 0x0};
		char** argv2 = (char**) malloc(sizeof(char*) * 2);
		int argc = 1;
		argv2[0] = argv[0];
		argv2[1] = argv[1];
		dbg::out(dbg::info, "mpi") << "Initializing MPI..." << std::endl;
		_env = boost::shared_ptr<mpi_env_t>(new mpi_env_t(argc, argv2, true));
		dbg::out(dbg::info, "mpi") << "MPI initialized" << std::endl;
		_world = boost::shared_ptr<mpi_com_t>(new mpi_com_t());

		nr_of_bins = Params::cmoea::nb_of_bins;
		bins_total = nr_of_bins * replicates;

		// Lower bound on the number of bins per instance
		_min_bins_inst = bins_total / _world->size();

		// Upper bound on the number of bins per instance
		_max_bins_inst = _min_bins_inst + 1;

		// The number of bins remaining after each instance is assigned its
		// bins_per_instance
		_overfull_inst = bins_total % _world->size();

		// The total number of bins assigned to over-full instances
		_overfull_bins = _max_bins_inst * _overfull_inst;

		// The number of bins this instance has to handle
		int my_nr_bins;
		if(_world->rank() < _overfull_inst){
			my_nr_bins = _max_bins_inst;
			_first_bin_id = _world->rank() * _max_bins_inst;
		} else {
			my_nr_bins = _min_bins_inst;
			int other_instances = (_world->rank()-_overfull_inst);
			int other_bins = other_instances * _min_bins_inst;
			_first_bin_id = _overfull_bins + other_bins;
		}

		// Create the necessary bins
		for( size_t i=0; i<my_nr_bins; ++i){
			_bins.push_back(pop_t());
		}

		// Print parameters
		DBO << "Objectives: " << nr_of_bins << dbe;
		DBO << "Min bins per instance: " << _min_bins_inst << dbe;
		DBO << "Max bins per instance: " << _max_bins_inst << dbe;
		DBO << "Over-full instances: " << _overfull_inst << dbe;
		DBO << "Over-full bins: " << _overfull_bins << dbe;
		DBO << "First bin id: " << _first_bin_id << dbe;
		DBO << "My bins: " << _bins.size() << dbe;
    }

    int get_rank(){
    	return _world->rank();
    }

    /**
     * Creates a random initial population.
     *
     * The number of generated individuals is determined by
     * init_batch * init_size, but the final population size is determined by
     * bin_size.
     */
    void random_pop(){
        dbg::trace trace("ea", DBG_HERE);
        parallel::init();

        // Clear the population
        this->_pop.clear();

        // Define bins
        pop_t send_pop;
        pop_t recv_pop;
        bins_t new_indiv_bins (_bins.size());

        // Calculate the number of batches
        unsigned nr_of_batches = (init_size / max_batch_size);
        unsigned remainder = init_size % max_batch_size;
        if(remainder) ++nr_of_batches;

        // Generate and evaluate new random individuals
        for(size_t bin_i=0; bin_i<_bins.size(); ++bin_i){
        	DBO << "Pop for bin: " << bin_i << dbe;
        	for(unsigned j=0; j<nr_of_batches; ++j){
        		// Calculate batch size
            	unsigned batch_size = max_batch_size;
            	if(j==(nr_of_batches-1) && remainder > 0){
            		batch_size = remainder;
            	}

        		// Generate the new random individuals
        		new_indiv_bins[bin_i].resize(batch_size);
        		int i = 0;
        		BOOST_FOREACH(phen_ptr_t& indiv, new_indiv_bins[bin_i]){
        			DBO << "Random indiv: " << i++ << dbe;
        			indiv = phen_ptr_t(new phen_t());
        			indiv->random();
        		}

            	// Evaluate the new individuals
        		DBO << "Evaluating pop" << std::endl;
            	_evaluate(new_indiv_bins[bin_i]);
        	}
        }

        // Exchange individuals
        for(size_t offset=0; offset<bin_size; offset+=max_send_size){
        	send_pop.clear();
        	recv_pop.clear();
        	for(size_t bin_i=0; bin_i<_bins.size(); ++bin_i){
        		pop_it_t it_slice_begin = new_indiv_bins[bin_i].begin();
        		pop_it_t it_end = new_indiv_bins[bin_i].end();
        		std::advance (it_slice_begin, offset);
        		pop_it_t it_slice_end = it_slice_begin;
        		for(size_t i=0; i<max_send_size && it_slice_end != it_end; ++i){
        			++it_slice_end;
        		}
        		send_pop.insert(send_pop.end(), it_slice_begin, it_slice_end);
        	}

        	g_exchange_v(_world, send_pop, recv_pop);

        	// Add all individuals to the same population and perform selection
        	for(size_t bin_i=0; bin_i<_bins.size(); ++bin_i){
        		this->_pop.clear();

            	// Copy the current individuals to the population for processing
            	compare::merge(this->_pop, _bins[bin_i]);

        		// Copy the new individuals to the population for processing
        		compare::merge(this->_pop, send_pop);

        		// Add the received individuals to the population for processing
        		compare::merge(this->_pop, recv_pop);

        		// Perform selection on the current population
        		_survivor_selection(bin_i);

        		// Copy the current population back to the appropriate bin
        		_copy_to_bin(bin_i, this->_pop);
        	}
        }

        // Copy everything back to the current population for serialization
        _copy_all_to_pop(this->_pop);
    }

    /**
     * Runs one epoch of our evolutionary algorithm.
     */
    void epoch(){
        dbg::trace trace("ea", DBG_HERE);
        DBO << "Starting epoch: " << this->_gen << dbe;

        // Clear the population
        this->_pop.clear();

        // Define bins
        pop_t send_pop;
        pop_t recv_pop;
        bins_t new_indiv_bins (_bins.size());

        // Generate and evaluate new individuals
        for(size_t bin_i=0; bin_i<_bins.size(); ++bin_i){
        	for (size_t i = 0; i < (offspring_per_generation/2); ++i){
        		DBO << "New indiv: " << i << dbe;
        		phen_ptr_t p1 = selection(_bins[bin_i]);
        		phen_ptr_t p2 = selection(_bins[bin_i]);
        		phen_ptr_t i1, i2;
        		p1->cross(p2, i1, i2);
        		i1->mutate();
        		i2->mutate();
        		new_indiv_bins[bin_i].push_back(i1);
        		new_indiv_bins[bin_i].push_back(i2);
        	}

        	// Evaluate the new individuals
        	_evaluate(new_indiv_bins[bin_i]);
        }

#ifdef EA_EVAL_ALL
        for(size_t bin_i=0; bin_i<_bins.size(); ++bin_i){
        	_evaluate(_bins[bin_i]);
        }
#endif


//        // Decide which individuals to send
//        for(size_t bin_i=0; bin_i<_bins.size(); ++bin_i){
//
//        	// Send the current (old) population
//        	//compare::merge(send_bins[bin_i], _bins[bin_i]);
//
//        	// Send the newly created individuals
//        	compare::merge(send_bins[bin_i], new_indiv_bins[bin_i]);
//        }


        /** OLD MP-CMOEA EXCHANGE START **/
//        // Get the set of all partners that we need to interact with
//        std::set<int> partners;
//        std::map<int, std::vector<int> > recv_bin_map;
//        std::map<int, std::vector<int> > send_bin_map;
//        std::map<int, int> recv_map;
//        for(size_t bin_i=0; bin_i<_bins.size(); ++bin_i){
//        	_get_partners(bin_i, partners, recv_bin_map, send_bin_map, recv_map);
//        }
//
//#if defined(DBG_ENABLED)
//        {
//        	DBO_MPI << "Partners of instance: " << _world->rank() << dbe;
//        	std::set<int>::iterator it;
//        	for (it = partners.begin(); it != partners.end(); ++it){
//        		DBO_MPI << "  partner: " << *it << dbe;
//        	}
//        }
//#endif
//        {
//        	std::set<int>::iterator it;
//        	for (it = partners.begin(); it != partners.end(); ++it){
//        		_exchange_send(*it, recv_bin_map, send_bin_map, send_bins, recv_bins, recv_map);
//        	}
//        	for (it = partners.begin(); it != partners.end(); ++it){
//        		_exchange_recv(*it, recv_bin_map, send_bin_map, send_bins, recv_bins, recv_map);
//        	}
//        }
//        _wait_all();
		/** OLD MP-CMOEA EXCHANGE END **/

//        // Exchange individuals with partners in the correct order
//        while(!partners.empty()){
//        	int partner = *partners.begin();
//        	partners.erase(partner);
//        	_exchange(partner, recv_bin_map, send_bin_map, send_bins, recv_bins, recv_map);
//        }


        // Exchange individuals
        for(size_t bin_i=0; bin_i<_bins.size(); ++bin_i){
        	compare::merge(send_pop, new_indiv_bins[bin_i]);
        }

        g_exchange_v(_world, send_pop, recv_pop);

        // Add all individuals to the same population and perform selection
        for(size_t bin_i=0; bin_i<_bins.size(); ++bin_i){
        	this->_pop.clear();

        	// Copy the current individuals to the population for processing
        	compare::merge(this->_pop, _bins[bin_i]);

        	// Copy the new individuals to the population for processing
        	compare::merge(this->_pop, send_pop);

        	// Add the received individuals to the population for processing
        	compare::merge(this->_pop, recv_pop);

        	// Perform selection on the current population
        	_survivor_selection(bin_i);

        	// Copy the current population back to the appropriate bin
        	_copy_to_bin(bin_i, this->_pop);
        }

        // Copy everything back to the current population for serialization
        _copy_all_to_pop(this->_pop);

        DBO << "Epoch " << this->_gen << " complete." << dbe;
    }

    void barrier(){
    	_world->barrier();
    }

protected:
    // Pointer to the MPI environment
    boost::shared_ptr<mpi_env_t> _env;

    // Pointer to the MPI world
    boost::shared_ptr<mpi_com_t> _world;

    std::vector<boost::mpi::packed_oarchive*> _send_buffer;
    std::vector<size_t*> _send_buffer_sizes;
    std::vector<MPI_Request*> _outstanding_requests;

    // Vector of bins stores all bins governed by this instance.
    bins_t _bins;

    // The global index of the first bin this instance is responsible for
    int _first_bin_id;

    // The lower bound on the number of bins handled by any one instance
    int _min_bins_inst;

    // The upper bound on the number of bins handled by any one instance
    // Because bins are distributed as evenly over instances as possible, will
    // always be _min_bins_inst + 1
    int _max_bins_inst;

    // The number of "over-full instances" (MPI instances responsible for
    // _max_bins_inst bins).
    int _overfull_inst;

    // The total number of bins allocated to "over-full instances"
    int _overfull_bins;

    /**
     * Evaluates all individuals in the provided population.
     *
     * @param pop: The population to evaluate.
     */
    void _evaluate(pop_t& pop){
    	dbg::trace trace("ea", DBG_HERE);
    	this->_eval.eval(pop, 0, pop.size(), this->_fit_proto);
#if defined(DBG_ENABLED)
    	for(size_t i=0; i<pop.size(); ++i){
    		DBOF << "Fit indiv: " << i << dbe;
    		for(size_t j=0; j<nr_of_bins; ++j){
    			DBOF << "  bin " << j <<
    					": " << pop[i]->fit().getBinFitness(j) << dbe;
    		}
    	}
#endif
    }

    /**
     * Performs survivor selection on the bin at the provided index.
     *
     * @param bin_i: The local bin index of the bin to evaluate.
     */
    void _survivor_selection(size_t bin_i){
    	dbg::trace trace("ea", DBG_HERE);

    	// Determine the type of bin that we govern based on our rank
    	size_t bin_obj = (_first_bin_id + bin_i) % nr_of_bins;
    	DBO << "Performing selection for bin: " << bin_obj << dbe;
    	_cat_to_obj(this->_pop, bin_obj);

    	// Apply modifiers
    	DBO << "  Applying modifiers" << dbe;
    	this->apply_modifier();

#if defined(DBG_ENABLED)
    	for(size_t i=0; i<this->_pop.size(); ++i){
    		DBOO << "  Pop before selection: " << i << dbe;
    		for(size_t j=0; j<this->_pop[i]->fit().objs().size(); ++j){
    			DBOO << "    obj " << j <<
    					": " << this->_pop[i]->fit().objs()[j] << dbe;
    		}
    	}
#endif

    	// Perform selection
    	DBO << "  Performing selection" << dbe;
    	pop_t ptmp;
    	sort_f()(this->_pop, ptmp, bin_size);
    	this->_pop = ptmp;

#if defined(DBG_ENABLED)
    	for(size_t i=0; i<this->_pop.size(); ++i){
    		DBOO << "  Pop after selection: " << i << dbe;
    		for(size_t j=0; j<this->_pop[i]->fit().objs().size(); ++j){
    			DBOO << "    obj " << j <<
    					": " << this->_pop[i]->fit().objs()[j] << dbe;
    		}
    	}
#endif

    }


    /**
     * Initialize the current bins given a population.
     *
     * @param pop: The population with which to initialize the current bins.
     */
    void _set_pop(const pop_t& pop) {
        dbg::trace trace("ea", DBG_HERE);
        size_t size_expected = _bins.size()*bin_size;
        DBO << "Pop size from archive: " << pop.size() << dbe;
        DBO << "Pop size expected: " << size_expected << dbe;
        dbg::assertion(DBG_ASSERTION(pop.size() == size_expected));
        size_t k=0;
        for(size_t i=0; i<_bins.size(); ++i){
        	for(size_t j=0; j<bin_size; ++j){
        		_bins[i].push_back(pop[k]);
        		++k;
        	}
        }
    }

    /**
     * Copy all individuals from the provided population to the bin at the
     * provided bin index.
     *
     * @param bin_i: The local bin index of the bin to copy to.
     * @param pop: The population of individuals to copy to that bin.
     */
    void _copy_to_bin(size_t bin_i, const pop_t& pop){
    	dbg::trace trace("ea", DBG_HERE);
    	_bins[bin_i].clear();
    	compare::merge(_bins[bin_i], pop);
    }

    /**
     * Copy all individuals of all bins to the provided population.
     *
     * @param pop (out): The population to which to copy the current bins.
     */
    void _copy_all_to_pop(pop_t& pop){
    	dbg::trace trace("ea", DBG_HERE);
    	pop.clear();
    	for(size_t bin_i=0; bin_i<_bins.size(); ++bin_i){
    		compare::merge(pop, _bins[bin_i]);
    	}
    }


    /**
     * Copy the performance on the task associated with the provided bin index
     * to one of the current objectives of the all individuals of the provided
     * population.
     *
     * @param pop: The population of individuals for which to copy task
     *     performance.
     * @param bin_i: The global bin index of the task performance that needs to
     *     be copied.
     */
    void _cat_to_obj(pop_t& pop, size_t bin_i){
        dbg::trace trace("ea", DBG_HERE);
        for(size_t i=0; i<pop.size(); ++i){
        	pop[i]->fit().set_obj(obj_index, pop[i]->fit().getBinFitness(bin_i));
        }
    }

    /**
     * Calculates which other MPI instances we need to communicate with, and
     * what data needs to be exchanged.
     *
     * @param local_bin_i: The local bin index for which we want to exchange
     *     information.
     * @param partners (out): Reference to the set of partners (mpi instances)
     *     with whom we need to communicate.
     * @param recv_bin_map (out): Reference to a map mapping instance rank to
     *     a vector of global bin indices of bins that we will receive from that
     *     instance.
     * @param send_bin_map (out): Reference to a map mapping instance rank to a
     *     vector of global bin indices that we will send to that instance.
     * @param recv_map (out): Map indicating which of the received bins should
     *     be added to which of the bins governed by this instance.
     */
    void _get_partners(int local_bin_i,
    		std::set<int>& partners,
			inst_bin_map_t& recv_bin_map,
			inst_bin_map_t& send_bin_map,
			std::map<int, int>& recv_map){
    	dbg::trace trace("ea", DBG_HERE);

    	// Determine with which bins to exchange individuals
    	int global_bin_i = _get_global_bin_i(local_bin_i);
    	int offset = (this->_gen % (bins_total-1)) + 1;
    	int bin_id1 =  compare::mod(global_bin_i + offset, bins_total);
    	int bin_id2 =  compare::mod(global_bin_i - offset, bins_total);
    	int inst1 = _get_instance_from_bin(bin_id1);
    	int inst2 = _get_instance_from_bin(bin_id2);

    	send_bin_map[inst1].push_back(global_bin_i);
    	send_bin_map[inst2].push_back(global_bin_i);
    	recv_bin_map[inst1].push_back(bin_id1);
    	recv_bin_map[inst2].push_back(bin_id2);
    	recv_map[bin_id1] = global_bin_i;
    	recv_map[bin_id2] = global_bin_i;

    	partners.insert(inst1);
    	partners.insert(inst2);
    }

    /**
     * Exchange all necessary bins with the indicated partner.
     *
     * @param partner: Rank of the MPI instance with whom to exchange data.
     * @param recv_bin_map: The map of MPI instances and bins to receive.
     * @param send_bin_map: The map on MPI instances and bins to send.
     * @param send_bins: Vector containing the actual bins to send.
     * @param recv_bins (out): Vector for storing received bins.
     * @param recv_map: Map indicating which received bins should be added to
     *     which of the local bins (in global bin ids).
     */
    void _exchange(int partner,
    		inst_bin_map_t& recv_bin_map,
			inst_bin_map_t& send_bin_map,
			const bins_t& send_bins,
			bins_t& recv_bins,
			const std::map<int, int>& recv_map){
    	dbg::trace trace("ea", DBG_HERE);
    	DBO_MPI << "Exchanging with partner: " << partner << dbe;

    	// These are the bins I am going to send, and in the order in which I
    	// will send them
    	std::vector<int> bins_to_send = send_bin_map[partner];
    	compare::sort(bins_to_send);

    	// These are the bins I am going to receive, and in the order in which
    	// I will receive them
    	std::vector<int> bins_to_receive = recv_bin_map[partner];
    	compare::sort(bins_to_receive);

    	if(_world->rank() == partner){
    		DBO_MPI << "Performing local exchange" << dbe;
    		for(size_t i=0; i<bins_to_send.size(); ++i){
    			int send_bin_i = _get_local_bin_i(bins_to_send[i]);
    			int recv_bin_i = _get_local_bin_i(bins_to_receive[i]);
    			recv_bins[recv_bin_i] = send_bins[send_bin_i];
    		}
    	} else if(_world->rank() < partner){
    		DBO_MPI << "My rank is lower, sending first" << dbe;
    		for(size_t i=0; i<bins_to_send.size(); ++i){
    			DBO_MPI << "Sending bin: " << bins_to_send[i] << dbe;
    			int local_bin_i = _get_local_bin_i(bins_to_send[i]);
    			DBO_MPI << "Local bin index: " << local_bin_i << dbe;
    			_send(partner, send_bins[local_bin_i]);
    		}
    		for(size_t i=0; i<bins_to_receive.size(); ++i){
    			DBO_MPI << "Receiving bin: " << bins_to_receive[i] << dbe;
    			int global_bin_i = recv_map.at(bins_to_receive[i]);
    			DBO_MPI << "Adding it to bin: " << global_bin_i << dbe;
    			int local_bin_i = _get_local_bin_i(global_bin_i);
    			DBO_MPI << "Local bin index: " << local_bin_i << dbe;
    			_recv(partner, recv_bins[local_bin_i]);
    		}
    	} else {
    		DBO_MPI << "My rank is higher, receiving first" << dbe;
    		for(size_t i=0; i<bins_to_receive.size(); ++i){
    			DBO_MPI << "Receiving bin: " << bins_to_receive[i] << dbe;
    			int global_bin_i = recv_map.at(bins_to_receive[i]);
    			DBO_MPI << "Adding it to bin: " << global_bin_i << dbe;
    			int local_bin_i = _get_local_bin_i(global_bin_i);
    			DBO_MPI << "Local bin index: " << local_bin_i << dbe;
    			_recv(partner, recv_bins[local_bin_i]);
    		}
    		for(size_t i=0; i<bins_to_send.size(); ++i){
    			DBO_MPI << "Sending bin: " << bins_to_send[i] << dbe;
    			int local_bin_i = _get_local_bin_i(bins_to_send[i]);
    			DBO_MPI << "Local bin index: " << local_bin_i << dbe;
    			_send(partner, send_bins[local_bin_i]);
    		}
    	}
    }

    /**
     * Exchange all necessary bins with the indicated partner.
     *
     * @param partner: Rank of the MPI instance with whom to exchange data.
     * @param recv_bin_map: The map of MPI instances and bins to receive.
     * @param send_bin_map: The map on MPI instances and bins to send.
     * @param send_bins: Vector containing the actual bins to send.
     * @param recv_bins (out): Vector for storing received bins.
     * @param recv_map: Map indicating which received bins should be added to
     *     which of the local bins (in global bin ids).
     */
    void _exchange_send(int partner,
    		inst_bin_map_t& recv_bin_map,
			inst_bin_map_t& send_bin_map,
			const bins_t& send_bins,
			bins_t& recv_bins,
			const std::map<int, int>& recv_map){
    	dbg::trace trace("ea", DBG_HERE);
    	DBO_MPI << "Sending to partner: " << partner << dbe;

    	// These are the bins I am going to send, and in the order in which I
    	// will send them
    	std::vector<int> bins_to_send = send_bin_map[partner];
    	compare::sort(bins_to_send);

    	// These are the bins I am going to receive, and in the order in which
    	// I will receive them
    	std::vector<int> bins_to_receive = recv_bin_map[partner];
    	compare::sort(bins_to_receive);

    	if(_world->rank() == partner){
    		DBO_MPI << "Performing local exchange" << dbe;
    		for(size_t i=0; i<bins_to_send.size(); ++i){
    			int send_bin_i = _get_local_bin_i(bins_to_send[i]);
    			int recv_bin_i = _get_local_bin_i(bins_to_receive[i]);
    			recv_bins[recv_bin_i] = send_bins[send_bin_i];
    		}
    	} else {
    		DBO_MPI << "Sending:" << dbe;
    		for(size_t i=0; i<bins_to_send.size(); ++i){
    			DBO_MPI << "Sending bin: " << bins_to_send[i] << dbe;
    			int local_bin_i = _get_local_bin_i(bins_to_send[i]);
    			DBO_MPI << "Local bin index: " << local_bin_i << dbe;
    			_sendi(partner, send_bins[local_bin_i], int(i));
    		}
    	}
    }

    /**
     * Exchange all necessary bins with the indicated partner.
     *
     * @param partner: Rank of the MPI instance with whom to exchange data.
     * @param recv_bin_map: The map of MPI instances and bins to receive.
     * @param send_bin_map: The map on MPI instances and bins to send.
     * @param send_bins: Vector containing the actual bins to send.
     * @param recv_bins (out): Vector for storing received bins.
     * @param recv_map: Map indicating which received bins should be added to
     *     which of the local bins (in global bin ids).
     */
    void _exchange_recv(int partner,
    		inst_bin_map_t& recv_bin_map,
			inst_bin_map_t& send_bin_map,
			const bins_t& send_bins,
			bins_t& recv_bins,
			const std::map<int, int>& recv_map){
    	dbg::trace trace("ea", DBG_HERE);
    	DBO_MPI << "Receiving from partner: " << partner << dbe;

    	// These are the bins I am going to send, and in the order in which I
    	// will send them
    	std::vector<int> bins_to_send = send_bin_map[partner];
    	compare::sort(bins_to_send);

    	// These are the bins I am going to receive, and in the order in which
    	// I will receive them
    	std::vector<int> bins_to_receive = recv_bin_map[partner];
    	compare::sort(bins_to_receive);


    	if(_world->rank() == partner){
    		DBO_MPI << "Partner is local: exchange already completed" << dbe;
    	} else {
    		DBO_MPI << "Receiving" << dbe;
    		for(size_t i=0; i<bins_to_receive.size(); ++i){
    			DBO_MPI << "Receiving bin: " << bins_to_receive[i] << dbe;
    			int global_bin_i = recv_map.at(bins_to_receive[i]);
    			DBO_MPI << "Adding it to bin: " << global_bin_i << dbe;
    			int local_bin_i = _get_local_bin_i(global_bin_i);
    			DBO_MPI << "Local bin index: " << local_bin_i << dbe;
    			_recv(partner, recv_bins[local_bin_i], int(i));
    		}
    	}
    }

    /**
     * Send the supplied population to the indicated MPI instance.
     *
     * @param partner: The rank of the MPI instance to send to.
     * @param pop: The population to send.
     */
    void _sendi(int partner, const pop_t& pop, int tag=1){
    	dbg::trace trace("ea", DBG_HERE);
    	MPI_Request* req1 = new MPI_Request();
    	MPI_Request* req2 = new MPI_Request();
    	_outstanding_requests.push_back(req1);
    	_outstanding_requests.push_back(req2);
        std::vector<MPI_Request> req(2);
        DBO_MPI << "sending to rank: " << partner << std::endl;

        // Serializing population

        boost::mpi::packed_oarchive* oa = new boost::mpi::packed_oarchive (*_world);
        _send_buffer.push_back(oa);

		//boost::mpi::packed_oarchive oa(*_world);
        (*oa) << pop;
        //_outstanding_requests.push_back(MPI_Request());
        size_t* size_p = new size_t((*oa).size());
        _send_buffer_sizes.push_back(size_p);

        // Send the the number of individuals that will be send
        const void* size_1 = size_p;
        BOOST_MPI_CHECK_RESULT(MPI_Isend,
                (const_cast<void*>(size_1), 1,
                 boost::mpi::get_mpi_datatype<std::size_t>(*size_p),
				 partner, tag, *_world, req1));

        // Actually send the individuals
        BOOST_MPI_CHECK_RESULT(MPI_Isend,
                (const_cast<void*>((*oa).address()),
                (*oa).size(),
                 MPI_PACKED,
				 partner, tag, *_world, req2));

        // Wait until everything is send successfully
        //BOOST_MPI_CHECK_RESULT(MPI_Waitall, (req.siz(), &req[0], MPI_STATUSES_IGNORE));
    }

    /**
     * Send the supplied population to the indicated MPI instance.
@@ -640,29 +748,53 @@ protected:
     */
    void _send(int partner, const pop_t& pop){
       dbg::trace trace("ea", DBG_HERE);
        std::vector<MPI_Request> req(2);
        DBO_MPI << "sending to rank: " << partner << std::endl;
        int tag = 1;

        // Serializing population
        boost::mpi::packed_oarchive oa(*_world);
        oa << pop;

        // Send the the number of individuals that will be send
        const void* size_1 = &oa.size();

        BOOST_MPI_CHECK_RESULT(MPI_Isend,
                (const_cast<void*>(size_1), 1,
                 boost::mpi::get_mpi_datatype<std::size_t>(oa.size()),
                                partner, tag, *_world, &req[0]));

        // Actually send the individuals
        BOOST_MPI_CHECK_RESULT(MPI_Isend,
                (const_cast<void*>(oa.address()), oa.size(),
                 MPI_PACKED, partner, tag, *_world, &req[1]));


        // Wait until everything is send successfully
        BOOST_MPI_CHECK_RESULT(MPI_Waitall, (2, &req[0], MPI_STATUSES_IGNORE));
    }

    void _wait_all(){
    	std::vector<MPI_Request> requests;
    	for(size_t i=0; i<_outstanding_requests.size(); ++i){
    		requests.push_back(*(_outstanding_requests[i]));
    	}
    	BOOST_MPI_CHECK_RESULT(MPI_Waitall, (requests.size(), &requests[0], MPI_STATUSES_IGNORE));
    	for(size_t i=0; i<_outstanding_requests.size(); ++i){
    		delete _outstanding_requests[i];
    	}
    	_outstanding_requests.clear();
    	for(size_t i=0; i<_send_buffer_sizes.size(); ++i){
    		delete _send_buffer_sizes[i];
    	}
    	_send_buffer_sizes.clear();
    	for(size_t i=0; i<_send_buffer.size(); ++i){
    		delete _send_buffer[i];
    	}
    	_send_buffer.clear();
    }


    /**
     * Receive individuals from the indicated MPI instance.
     *
     * @param partner: The rank of the MPI instance to receive from.
     * @param pop (out): Reference to a population in which to store the
     *     received individuals.
     */
    void _recv(int partner, pop_t& pop, int tag = 1){
    	dbg::trace trace("ea", DBG_HERE);
        // Receive data from the root.
        MPI_Status stat;
        std::size_t count;
        int recv = 0;
        using namespace boost::mpi;

        DBO_MPI << "receiving from rank: " << partner << dbe;

        BOOST_MPI_CHECK_RESULT(MPI_Recv,
                (&count, 1,  get_mpi_datatype<std::size_t>(count),
                partner, tag, *_world, &stat));
        MPI_Get_count(&stat,  get_mpi_datatype<std::size_t>(count), &recv);
        DBO_MPI  <<
				" received size_t of size: " << recv <<
				" from " << stat.MPI_SOURCE <<
				" with tag " << stat.MPI_TAG << std::endl;

        // Prepare input buffer and receive the message
        mpi_iarchive_t ia(*_world);
        ia.resize(count);
        BOOST_MPI_CHECK_RESULT(MPI_Recv,
                (ia.address(), ia.size(), MPI_PACKED,
                partner, tag,
                 *_world, &stat));
        MPI_Get_count(&stat, MPI_PACKED, &recv);
        DBO_MPI <<
				" received archive of size: " << recv <<
				" from " << stat.MPI_SOURCE <<
				" with tag " << stat.MPI_TAG << std::endl;

        // De-serializing population
        ia >> pop;
    }


    /**
     * Takes a global bin index and returns the local bin index.
     *
     * @param global_bin_i: The global bin index to convert.
     * @return: The local bin index.
     */
    inline int _get_local_bin_i(int global_bin_i) const{
    	dbg::trace trace("ea", DBG_HERE);
    	dbg::assertion(DBG_ASSERTION(global_bin_i >= 0));
    	dbg::assertion(DBG_ASSERTION(global_bin_i < bins_total));
    	int result = global_bin_i - _first_bin_id;
    	dbg::assertion(DBG_ASSERTION(result >= 0));
    	dbg::assertion(DBG_ASSERTION(result < _bins.size()));
    	return result;
    }

    /**
     * Takes a local bin index and returns its global bin index.
     *
     * @param local_bin_i: The local bin index to convert.
     * @return: The global bin index.
     */
    inline int _get_global_bin_i(int local_bin_i) const{
    	dbg::trace trace("ea", DBG_HERE);
    	dbg::assertion(DBG_ASSERTION(local_bin_i >= 0));
    	dbg::assertion(DBG_ASSERTION(local_bin_i < _bins.size()));
    	int result = local_bin_i + _first_bin_id;
    	dbg::assertion(DBG_ASSERTION(result >= 0));
    	dbg::assertion(DBG_ASSERTION(result < bins_total));
    	return result;
    }


    /**
     * Returns the rank of the instance associated with the supplied bin_id.
     *
     * @param bin_id: The global bin index for which we want to retrieve the MPI
     *     instance.
     * @return: The rank of the MPI instance responsible for the provided bin
     *     index.
     */
    inline int _get_instance_from_bin(int bin_id) const{
    	dbg::trace trace("ea", DBG_HERE);
    	int instance_rank;

		if(bin_id < _overfull_bins){
			// The instance we are looking for is one of the over-full instances
			instance_rank = bin_id / _max_bins_inst;
		} else{
			// The instance we are looking for is one of the other instances
			int other_bins = (bin_id - _overfull_bins);
			int other_instances = other_bins / _min_bins_inst;
			instance_rank = _overfull_inst + other_instances;
		}

		return instance_rank;
    }
};
}
}

#undef DBO
#undef DBO_MPI
#undef DBOF
#undef DBOO

#endif /* MODULES_CMOEA_CMOEA_NSGA2_MPI_HPP_ */
