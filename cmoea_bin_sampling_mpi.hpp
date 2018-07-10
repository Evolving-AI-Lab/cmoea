/*
 * cmoea_bin_sampling_mpi.hpp
 *
 *  Created on: Jun 1, 2017
 *      Author: joost
 */

#ifndef MODULES_CMOEA_CMOEA_BIN_SAMPLING_MPI_HPP_
#define MODULES_CMOEA_CMOEA_BIN_SAMPLING_MPI_HPP_

#include "cmoea_bin_sampling.hpp"
#include "mpi_util.hpp"

// Debug defines
#define DBOW dbg::out(dbg::info, "mpi") << "Worker " << _world->rank()
#define DBOM dbg::out(dbg::info, "mpi") << "Master " << this->eval().world()->rank()
#define DBOE dbg::out(dbg::info, "ea")

namespace sferes
{
namespace ea
{

template<typename Phen, typename FitModifier, typename Params>
class CmoeaSelectSampleTask{
public:
	// Params

    // The index used to temporarily store the category
    // This index should hold a dummy value, as it will be overwritten
    // constantly.
    // The default would be 0.
    static const size_t obj_index = Params::cmoea::obj_index;

    // The number of different tasks that need to be performed
    size_t nr_of_objs;

    // The size of each bin
    static const size_t bin_size = Params::cmoea::bin_size;

    // Type definitions
    typedef CmoeaSelectSampleTask<Phen, FitModifier, Params> this_t;
    typedef Phen phen_t;
    typedef crowd::Indiv<phen_t> crowd_t;
    typedef boost::shared_ptr<crowd_t> indiv_t;
    typedef std::vector<indiv_t> pop_t;
    typedef std::vector<boost::shared_ptr<phen_t> > ea_pop_t;
    typedef std::vector<std::vector<indiv_t> > front_t;
    typedef std::vector<char> bitset_t;

    // The type of Pareto domination sort to use.
    typedef typename Params::ea::dom_sort_f dom_sort_f;

    // The type non dominated comparator to use
    typedef typename Params::cmoea_nsga::non_dom_f non_dom_f;

    // The exact type of non-dominated sort function to use
    typedef fill_dom_sort_f<indiv_t, dom_sort_f, non_dom_f> sort_f;

    // Modifier for calculating distance
    typedef typename boost::mpl::if_<
    				 boost::fusion::traits::is_sequence<FitModifier>,
                     FitModifier,
                     boost::fusion::vector<FitModifier>
    >::type modifier_t;

    modifier_t _fit_modifier;
    ea_pop_t _pop;

    CmoeaSelectSampleTask(){
        nr_of_objs = Params::cmoea::nr_of_objs;
    }

    void run(boost::shared_ptr<boost::mpi::communicator> _world,
    		boost::mpi::status s,
			boost::shared_ptr<boost::mpi::environment> env)
    {
        dbg::trace trace("ea", DBG_HERE);
        pop_t pop;
        pop_t archive;
        pop_t new_bin;
        indiv_t temp;
        bitset_t objs;


        DBOW << " receiving broadcast from world: " << _world << std::endl;
        better_broadcast(_world, pop);
        DBOW << " received pop of size: " << pop.size() << std::endl;

        while(true){
        	DBOW << " waiting for message in cmoea_nsga2_mpi.hpp" << std::endl;
            s = _world->probe();
            DBOW << " receveived message in cmoea_nsga2_mpi.hpp tag: " <<
            		s.tag() << " source: " << s.source()  << std::endl;
            if (s.tag() == env->max_tag()){
                break;
            }

            DBOW << " receiving objectives." << std::endl;
            _world->recv(0, s.tag(), objs);
            DBOW << " objectives received." << std::endl;

            DBOW << " receiving archive." << std::endl;
            _world->recv(0, s.tag(), archive);
            DBOW << " archive received." << std::endl;

            for(size_t j = 0; j < pop.size(); ++j){
                archive.push_back(pop[j]);
            }
            DBOW << " applying modifier" << std::endl;
            _apply_modifier(archive);
            _cat_to_obj(archive, objs);
            DBOW << " sorting" << std::endl;
        	sort_f()(archive, new_bin, bin_size);
            DBOW << " sending bin: " << s.tag() <<
            		" size: " << new_bin.size() << std::endl;
            _world->send(0, s.tag(), new_bin);
        }
    }

    // modifiers
   void apply_modifier()
   { boost::fusion::for_each(_fit_modifier, ApplyModifier_f<this_t>(*this)); }

   const ea_pop_t& pop() const { return _pop; };
   ea_pop_t& pop() { return _pop; };

protected:
    /**
     * Applies the modifier to the supplied population (vector of individuals).
     *
     * Note that this overwrites the this->_pop population.
     */
    void _apply_modifier(pop_t pop){
        dbg::trace trace("ea", DBG_HERE);
        convert_pop(pop, this->_pop);
        apply_modifier();
    }


    /**
     * For the specified category and array, copies the category score to the
     * obj_index (usually 1).
     */
    void _cat_to_obj(pop_t& pop, bitset_t objs){
        dbg::trace trace("ea", DBG_HERE);
        for(size_t i=0; i<pop.size(); ++i){
        	double fit = 1;
        	for(size_t obj_i=0; obj_i<nr_of_objs; ++obj_i){
        		if(objs[obj_i]){
        			fit *= pop[i]->fit().getCmoeaObj(obj_i);
        		}
        	}
        	pop[i]->fit().set_obj(obj_index, fit);
        }
    }
};


// Main class
SFERES_EA(CmoeaBinSamplingMpi, CmoeaBinSampling){
public:
    typedef Phen phen_t;
    typedef crowd::Indiv<phen_t> crowd_t;
    typedef boost::shared_ptr<crowd_t> indiv_t;
    typedef std::vector<indiv_t> pop_t;
    typedef CmoeaBinSampling<Phen, Eval, Stat, FitModifier, Params,
    		Exact> parent_t;

//    static const size_t max_bins = parent_t::max_bins;

#ifndef NOMPISELECT
    /**
     * Adds the new `population' (vector of individuals) to the archive by
     * adding every individual to every bin, and then running NSGA 2 (or pNSGA)
     * selection on every bin.
     */
    void add_to_archive(pop_t& pop){
        dbg::trace trace("ea", DBG_HERE);
        //Add everyone to the archive

        //Set new task
        for (size_t i = 1; i < this->eval().world()->size(); ++i){
            this->eval().world()->send(i, this->eval().env()->max_tag(), 1);
        }


        //Broadcast the new individuals to all workers
        DBOM << "Broadcasting population of size: " <<
        		pop.size() << " to world: " << this->eval().world() <<
				std::endl;
        better_broadcast(this->eval().world(), pop);


        size_t current = 0;
        std::vector<bool> done(this->max_bins);
        std::fill(done.begin(), done.end(), false);

        // first round
        size_t world_size = this->eval().world()->size();
        for (size_t i = 1; i < world_size && current < this->max_bins; ++i) {
        	DBOM << "Sending bin: " << current  <<
        			" to worker " << i << std::endl;
        	this->eval().world()->send(i, current, this->_obj_combs[current]);
            this->eval().world()->send(i, current, this->_array[current]);
            ++current;
        }

        // Subsequent rounds
        while (current < this->max_bins){
            boost::mpi::status s = this->eval().world()->probe();
            DBOM << "Receiving bin: " << s.tag() << std::endl;
            this->eval().world()->recv(s.source(), s.tag(), this->_array[s.tag()]);
            DBOM << "Received bin: " << s.tag() <<
            		" size: " << this->_array[s.tag()].size()  << std::endl;
            done[s.tag()] = true;
            DBOM << "Sending bin: " << current << std::endl;
            this->eval().world()->send(s.source(), current, this->_obj_combs[current]);
            this->eval().world()->send(s.source(), current, this->_array[current]);
            ++current;
        }

        // Join
        bool all_done = true;
        do{
        	DBOM << "joining..."<<std::endl;
            all_done = true;
            for (size_t i = 0; i < this->max_bins; ++i){
                if (!done[i]){
                    boost::mpi::status s = this->eval().world()->probe();
                    DBOM << "Receiving bin: " << s.tag() << std::endl;
                    this->eval().world()->recv(s.source(), s.tag(),
                    		this->_array[s.tag()]);
                    DBOM << "Received bin: " << s.tag() <<
                    		" size: " << this->_array[s.tag()].size()  << std::endl;
                    done[s.tag()] = true;
                    all_done = false;
                }
            }
        }
        while (!all_done);
    }
#endif
};
}
}

// Undefine everything
#undef DBOM
#undef DBOW
#undef DBOE

#endif /* MODULES_CMOEA_CMOEA_BIN_SAMPLING_MPI_HPP_ */
