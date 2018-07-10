/*
 * modules/cmoea/cmoea_nsga2_mpi.hpp
 *
 *  Created on: Mar 12, 2015
 *      Author: Joost Huizinga
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
#include "cmoea_nsga2.hpp"
#include "mpi_util.hpp"
#include "nsga_util.hpp"

// Debug defines
#define DBOW dbg::out(dbg::info, "mpi") << "Worker " << _world->rank()
#define DBOM dbg::out(dbg::info, "mpi") << "Master " << this->eval().world()->rank()
#define DBOE dbg::out(dbg::info, "ea")

namespace karma = boost::spirit::karma;

namespace sferes
{
namespace ea
{

template<typename Phen, typename FitModifier, typename Params>                                                       \
class ArchiveTask{
public:
    //Params
    typedef ArchiveTask<Phen, FitModifier, Params> this_t;

    //The type of Pareto domination sort to use.
    //Currently available types are:
    // - sferes::ea::dom_sort_basic_f  (defined in sferes/ea/dom_sort_basic.hpp)
    //   Sorts according to pareto dominance and will add individuals from the
    //   highest to the lowest layer, with crowding as a tie-breaker in the last
    //   layer to be added.
    // - sferes::ea::dom_sort_no_duplicates_f
    //   (defined in sferes/ea/dom_sort_no_duplicates.hpp)
    //   Same as dom_sort_basic, accept that, for each front, only one
    //   individual per pareto location is added. Other individuals at the same
    //   location will be bumped to the next layer.
    typedef typename Params::ea::dom_sort_f dom_sort_f;

    //The type non dominated comparator to use
    //Currently available types are:
    // - sferes::ea::_dom_sort_basic::non_dominated_f
    //   (defined in sferes/ea/dom_sort_basic.hpp)
    //   Regular comparisons based on dominance
    // - sferes::ea::cmoea_nsga::prob_dom_f<Params>
    //   Comparisons based on probabilistic sorting, where some objectives can
    //   be stronger than others.
    typedef typename Params::cmoea_nsga::non_dom_f non_dom_f;

    //The index used to temporarily store the category
    //This index should hold a dummy value, as it will be overwritten
    //constantly.
    //The default would be 0.
    static const size_t obj_index = Params::cmoea::obj_index;

    //The number of objectives (bins) used in the map
    size_t nr_of_bins;

    //The size of each bin
    static const size_t bin_size = Params::cmoea::bin_size;

    //The number of individuals initially generated to fill the archive
    //If equal to the bin_size, every initially generated individual is added
    //to every bin of every category.
    //The init_size has to be greater than or equal to the bin_size
    static const size_t init_size = Params::pop::init_size;

    //Not actually the size of the standing population
    //(which is bin_size*nr_of_bins)
    //but the number of individuals that are generated each epoch.
    //Because individuals are always produced in pairs,
    //pop_size has to be divisible by 2.
    static const size_t indiv_per_gen = Params::pop::select_size;

    //Modifier for calculating distance
    typedef typename boost::mpl::if_<boost::fusion::traits::is_sequence<FitModifier>,
                     FitModifier,
                     boost::fusion::vector<FitModifier> >::type modifier_t;

    typedef Phen phen_t;
    typedef crowd::Indiv<phen_t> crowd_t;
    typedef boost::shared_ptr<crowd_t> indiv_t;
    typedef typename std::vector<indiv_t> pop_t;
    typedef typename std::vector<boost::shared_ptr<phen_t> > ea_pop_t;
    typedef typename std::vector<std::vector<indiv_t> > front_t;

    modifier_t _fit_modifier;
    ea_pop_t _pop;

    ArchiveTask(){
    	nr_of_bins = Params::cmoea::nb_of_bins;
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

            DBOW << " receiving archive." << std::endl;
            _world->recv(0, s.tag(), archive);
            DBOW << " archive received." << std::endl;

            for(size_t j = 0; j < pop.size(); ++j){
                archive.push_back(pop[j]);
            }
            DBOW << " applying modifier" << std::endl;
            _apply_modifier(archive);
            _cat_to_obj(archive, s.tag());
            DBOW << " sorting" << std::endl;
            _fill_nondominated_sort(archive, new_bin);
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
     * Converts a population from array individuals to regular individuals.
     */
    void _convert_pop(const pop_t& pop1, ea_pop_t& pop2){
        dbg::trace trace("ea", DBG_HERE);
        pop2.resize(pop1.size());
        for (size_t i = 0; i < pop1.size(); ++i){
            pop2[i] = pop1[i];
        }
    }

    /**
     * Converts a population from regular individuals to crowd individuals.
     */
    void _convert_pop(const ea_pop_t& pop1, pop_t& pop2){
        dbg::trace trace("ea", DBG_HERE);
        pop2.resize(pop1.size());
        for (size_t i = 0; i < pop1.size(); ++i){
            pop2[i] = boost::shared_ptr<crowd_t>(new crowd_t(*pop1[i]));
        }
    }

    /**
     * Does not actually convert anything, merely copies the content from one
     * pop to the other pop.
     */
    void _convert_pop(const pop_t& pop1, pop_t& pop2){
        dbg::trace trace("ea", DBG_HERE);
        pop2.resize(pop1.size());
        for (size_t i = 0; i < pop1.size(); ++i){
            pop2[i] = pop1[i];
        }
    }
 

    /**
     * Applies the modifier to the supplied population (vector of individuals).
     *
     * Note that this overwrites the this->_pop population.
     */
    void _apply_modifier(pop_t pop){
        dbg::trace trace("ea", DBG_HERE);
        _convert_pop(pop, this->_pop);
        apply_modifier();
    }

    /**
     * Selects a random individual from the supplied population.
     */
    indiv_t _selection(const pop_t& pop){
        dbg::trace trace("ea", DBG_HERE);
        int x1 = misc::rand< int > (0, pop.size());
        dbg::check_bounds(dbg::error, 0, x1, pop.size(), DBG_HERE);
        return pop[x1];
    }

    /**
     * Takes a mixed population, sorts it according to Pareto dominance, and
     * generates a new population depending on the bin size.
     *
     * @Param mixed_pop The mixed population from which to select.
     *                  The mixed population must be larger than the bin_size
     *                  for selection to occur.
     * @Param new_pop   Output parameter. After execution, should contain a
     * 					number of individuals equal to the bin_size, selected
     * 					based on Pareto dominance first, crowding second.
     */
    void _fill_nondominated_sort(pop_t& mixed_pop, pop_t& new_pop)
    {
        dbg::trace trace("ea", DBG_HERE);
        dbg::out(dbg::info, "ea") << "Mixed pop size: " << mixed_pop.size() <<
        		" bin size: " << bin_size << std::endl;
        dbg::assertion(DBG_ASSERTION(mixed_pop.size()));
        dbg::assertion(DBG_ASSERTION(mixed_pop.size() >= bin_size));

        //Rank the population according to Pareto fronts
        front_t fronts;
        _rank_crowd(mixed_pop, fronts);

        //Add Pareto layers to the new population until the current layer no
        //longer fits
        new_pop.clear();
        size_t front_index = 0;
        while(fronts[front_index].size() + new_pop.size() < bin_size){
            new_pop.insert(new_pop.end(), fronts[front_index].begin(),
            		fronts[front_index].end());
            ++front_index;
        }

        // sort the last layer
        size_t size_remaining = bin_size - new_pop.size();
        if (size_remaining > 0){
            dbg::assertion(DBG_ASSERTION(front_index < fronts.size()));
            std::sort(fronts[front_index].begin(), fronts[front_index].end(),
            		crowd::compare_crowd());
            for (size_t k = 0; k < size_remaining; ++k){
                new_pop.push_back(fronts[front_index][k]);
            }
        }
        dbg::assertion(DBG_ASSERTION(new_pop.size() == bin_size));
    }

    // --- rank & crowd ---

    /**
     * Ranks and crowds a population.
     *
     * Takes a population and divides it based on objectives.
     *
     * @param pop    The population to be ranked.
     * @param fronts The resulting Pareto fronts will be stored here.
     */
    void _rank_crowd(pop_t& pop, front_t& fronts)
    {
        dbg::trace trace("ea", DBG_HERE);
        //Execute ranking based on dominance
        std::vector<size_t> ranks;
        dom_sort_f()(pop, fronts, non_dom_f(), ranks);

        //Why are we assigning a crowd score to every individual?
        parallel::p_for(parallel::range_t(0, fronts.size()),
        		crowd::assign_crowd<indiv_t >(fronts));
    }


    /**
     * For the specified category and array, copies the category score to the
     * obj_index (usually 1).
     */
    void _cat_to_obj(pop_t& bin, size_t bin_i){
        dbg::trace trace("ea", DBG_HERE);
        for(size_t i=0; i<bin.size(); ++i){
        	float fit = bin[i]->fit().getBinFitness(bin_i);
            bin[i]->fit().set_obj(obj_index, fit);
        }
    }

    /**
     * Copies the stored diversity back to the relevant objective.
     *
     * Does nothing when DIV is not defined
     */
    void _div_to_obj(pop_t& bin, size_t category){
        dbg::trace trace("ea", DBG_HERE);
#if defined(DIV)
        for(size_t i=0; i<bin.size(); ++i){
            size_t div_index = bin[i]->fit().objs().size() - 1;
            float div = bin[i]->fit().getBinDiversity(category);
            bin[i]->fit().set_obj(div_index, div);
        }
#endif
    }

    /**
     * Copies the calculated diversity to the individuals diversity array.
     *
     * Does nothing when DIV is not defined
     */
    void _obj_to_div(pop_t& bin, size_t category){
        dbg::trace trace("ea", DBG_HERE);
#if defined(DIV)
        for(size_t j = 0; j < bin.size(); ++j){
            bin[j]->fit().initDiv();
            size_t div_index = bin[j]->fit().objs().size() - 1;
            bin[j]->fit().setDiv(category, bin[j]->fit().obj(div_index));
        }
#endif
    }
};


// Main class
SFERES_EA(CmoeaNsga2Mpi, CmoeaNsga2){
public:
    //Params

    //The type of Pareto domination sort to use.
    //Currently available types are:
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

    //The type non dominated comparator to use
    //Currently available types are:
    // - sferes::ea::_dom_sort_basic::non_dominated_f
    //   (defined in sferes/ea/dom_sort_basic.hpp)
    //   Regular comparisons based on dominance
    // - sferes::ea::cmoea_nsga::prob_dom_f<Params>
    //   Comparisons based on probabilistic sorting, where some objectives can
    //   be stronger than others.
    typedef typename Params::cmoea_nsga::non_dom_f non_dom_f;

    //The index used to temporarily store the category
    //This index should hold a dummy value, as it will be overwritten
    //constantly. The default would be 0.
    static const size_t obj_index = Params::cmoea::obj_index;

    //The number of objectives (bins) used in the map
    size_t nr_of_bins;

    //The size of each bin
    static const size_t bin_size = Params::cmoea::bin_size;

    //The number of individuals initially generated to fill the archive
    //If equal to the bin_size, every initially generated individual is added
    //to every bin of every category.
    //The init_size has to be greater than or equal to the bin_size
    static const size_t init_size = Params::pop::init_size;

    // Very large initial populations may cause CMOEA to run out of memory.
    // To avoid this, you can add the initial populations in init_batch batches
    // of init_size.
//    static const size_t init_batch = Params::pop::init_batch;

    //Not actually the size of the standing population
    //(which is bin_size*nr_of_bins)
    //but the number of individuals that are generated each epoch.
    //Because individuals are always produced in pairs,
    //pop_size has to be divisible by 2.
    static const size_t indiv_per_gen = Params::pop::select_size;

    typedef Phen phen_t;
    typedef crowd::Indiv<phen_t> crowd_t;
    typedef boost::shared_ptr<crowd_t> indiv_t;
    typedef std::vector<indiv_t> pop_t;
    typedef boost::shared_ptr<phen_t> raw_indiv_t;
    typedef std::vector<raw_indiv_t> raw_pop_t;
    typedef typename std::vector<pop_t> front_t;
    typedef std::vector<pop_t> array_t;
    SFERES_EA_FRIEND(CmoeaNsga2Mpi);

    CmoeaNsga2Mpi()
    {
        dbg::trace trace("ea", DBG_HERE);
        //dbg::compile_assertion<pop_size%2 == 0>("Population size has to be
        //divisible by 2.");
        DBOE << "Objectives: " << nr_of_bins << std::endl;
        nr_of_bins = Params::cmoea::nb_of_bins;
        this->_array.resize(nr_of_bins);
    }

//    void random_pop()
//    {
//        dbg::trace trace("ea", DBG_HERE);
//        DBOE << "Generating random pop" << std::endl;
//        //Create and evaluate the initial random population
//        parallel::init();
//        for(unsigned j=0; j<init_batch; ++j){
//            pop_t pop;
//            pop.resize(init_size);
//            int i = 0;
//            BOOST_FOREACH(indiv_t& indiv, pop)
//            {
//            	DBOE << "Creating random individual: " << i++ << std::endl;
//                indiv = indiv_t(new crowd_t());
//                indiv->random();
//            }
//            DBOE << "Evaluating population" << std::endl;
//            this->_eval.eval(pop, 0, pop.size(), this->_fit_proto);
//
//            DBOE << "Applying modifier" << std::endl;
//            _apply_modifier(pop);
//
//            DBOE << "Adding to archive" << std::endl;
//            _add_to_archive(pop);
//        }
//        DBOE << "Generating random pop done" << std::endl;
//
//        DBG_CONDITIONAL(dbg::info, "archive", this->_init_debug_array());
//    }
//
//    void epoch(){
//        dbg::trace trace("ea", DBG_HERE);
//
//        //We are creating and selecting a number of individuals equal to the
//        //population size. A simpler variant would only select and mutate one
//        //individual
//        pop_t ptmp;
//        for(size_t i=0; i<_array.size(); ++i){
//            if(_array[i].size() != bin_size){
//                std::cout << "Before reproduction: bin " << i <<
//                		" contains only " << _array[i].size() <<
//						" indiv." <<std::endl;
//            }
//        }
//        for (size_t i = 0; i < (indiv_per_gen/2); ++i)
//        {
//        	DBOE << "Creating individual: " << i <<std::endl;
//            indiv_t p1 = _selection(_array);
//            indiv_t p2 = _selection(_array);
//            indiv_t i1, i2;
//
//            p1->cross(p2, i1, i2);
//            i1->mutate();
//            i2->mutate();
//            ptmp.push_back(i1);
//            ptmp.push_back(i2);
//        }
//        this->_eval.eval(ptmp, 0, ptmp.size(), this->_fit_proto);
//        _add_to_archive(ptmp);
//
//        for(size_t i=0; i<_array.size(); ++i){
//            if(_array[i].size() != bin_size){
//                std::cout << "After reproduction: bin " << i <<
//                		" contains only " << _array[i].size() <<
//						" indiv." <<std::endl;
//            }
//        }
//
//        //For writing statistics only from the first bin:
//        _convert_pop(_array, this->_pop);
//
//        DBG_CONDITIONAL(dbg::info, "archive", this->_print_archive());
//    }
//
//    const array_t& archive() const { return _array; }


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
        std::vector<bool> done(nr_of_bins);
        std::fill(done.begin(), done.end(), false);

        // first round
        size_t world_size = this->eval().world()->size();
        for (size_t i = 1; i < world_size && current < nr_of_bins; ++i) {
        	DBOM << "Sending bin: " << current  <<
        			" to worker " << i << std::endl;
            this->eval().world()->send(i, current, this->_array[current]);
            ++current;
        }

        // Subsequent rounds
        while (current < nr_of_bins){
            boost::mpi::status s = this->eval().world()->probe();
            DBOM << "Receiving bin: " << s.tag() << std::endl;
            this->eval().world()->recv(s.source(), s.tag(), this->_array[s.tag()]);
            DBOM << "Received bin: " << s.tag() <<
            		" size: " << this->_array[s.tag()].size()  << std::endl;
            done[s.tag()] = true;
            DBOM << "Sending bin: " << current << std::endl;
            this->eval().world()->send(s.source(), current, this->_array[current]);
            ++current;
        }

        // Join
        bool all_done = true;
        do{
        	DBOM << "joining..."<<std::endl;
            all_done = true;
            for (size_t i = 0; i < nr_of_bins; ++i){
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

protected:
//    array_t _array;
//
//    /**
//     * Converts a population from array individuals to regular individuals.
//     */
//    void _convert_pop(const pop_t& pop1, raw_pop_t& pop2){
//        dbg::trace trace("ea", DBG_HERE);
//        pop2.resize(pop1.size());
//        for (size_t i = 0; i < pop1.size(); ++i){
//            pop2[i] = pop1[i];
//        }
//    }
//
//    /**
//     * Converts the entire array of individuals to regular individuals.
//     */
//    void _convert_pop(const array_t& array, raw_pop_t& pop2){
//        dbg::trace trace("ea", DBG_HERE);
//        pop2.resize(array.size() * bin_size);
//        size_t k=0;
//        for (size_t i = 0; i < array.size(); ++i){
//          for (size_t j = 0; j < bin_size; ++j){
//            pop2[k++] = array[i][j];
//          }
//        }
//    }
//
//    /**
//     * Converts a population from regular individuals to crowd individuals.
//     */
//    void _convert_pop(const raw_pop_t& pop1, pop_t& pop2){
//        dbg::trace trace("ea", DBG_HERE);
//        pop2.resize(pop1.size());
//        for (size_t i = 0; i < pop1.size(); ++i){
//            pop2[i] = boost::shared_ptr<crowd_t>(new crowd_t(*pop1[i]));
//        }
//    }
//
//    /**
//     * Initialize the archive give a population.
//     */
//    void _set_pop(const raw_pop_t& pop) {
//        dbg::trace trace("ea", DBG_HERE);
//        pop_t converted_pop;
//        this->_convert_pop(pop, converted_pop);
//
//        dbg::out(dbg::info, "continue") << "Adding: " << converted_pop.size()
//                << " to archive: " << this-> _array.size()
//                << " by " << this->bin_size << std::endl;
//        dbg::assertion(DBG_ASSERTION(pop.size() == converted_pop.size()));
//        dbg::assertion(DBG_ASSERTION(converted_pop.size() ==
//        		this->_array.size()*this->bin_size));
//
//        //Add everyone to the archive in the appropriate place
//        size_t pop_index = 0;
//        for(size_t i=0; i<this->_array.size(); ++i){
//            for(size_t j=0; j<this->bin_size; ++j){
//                this->_array[i].push_back(converted_pop[pop_index]);
//                ++pop_index;
//            }
//        }
//
//        DBG_CONDITIONAL(dbg::info, "archive", this->_init_debug_array());
//    }
//
//    /**
//     * Applies the modifier to the supplied population (vector of individuals).
//     *
//     * Note that this overwrites the this->_pop population.
//     */
//    void _apply_modifier(pop_t pop){
//        dbg::trace trace("ea", DBG_HERE);
//        _convert_pop(pop, this->_pop);
//        this->apply_modifier();
//    }
//
//
//
//    /**
//     * Selects a random individual from the supplied population.
//     */
//    indiv_t _selection(const pop_t& pop){
//        dbg::trace trace("ea", DBG_HERE);
//        int x1 = misc::rand< int > (0, pop.size());
//        dbg::check_bounds(dbg::error, 0, x1, pop.size(), DBG_HERE);
//        return pop[x1];
//    }
//
//    /**
//     * Selects a random individual from the supplied archive
//     */
//    indiv_t _selection(const array_t& archive){
//        dbg::trace trace("ea", DBG_HERE);
//        size_t category = misc::rand< size_t > (0, archive.size());
//        dbg::check_bounds(dbg::error, 0, category, archive.size(), DBG_HERE);
//        size_t size = archive[category].size();
//        size_t indiv_i = misc::rand< size_t > (0, size);
//        dbg::check_bounds(dbg::error, 0, indiv_i, size, DBG_HERE);
//        return archive[category][indiv_i];
//    }
//
//    /**
//     * Takes a mixed population, sorts it according to Pareto dominance, and
//     * generates a new population depending on the bin size.
//     *
//     * @Param mixed_pop The mixed population from which to select.
//     *                  The mixed population must be larger than the bin_size
//     *                  for selection to occur.
//     * @Param new_pop   Output parameter. After execution, should contain a
//     * 					number of individuals equal to the bin_size, selected
//     * 					based on Pareto dominance first, crowding second.
//     */
//    void _fill_nondominated_sort(pop_t& mixed_pop, pop_t& new_pop)
//    {
//        dbg::trace trace("ea", DBG_HERE);
//        dbg::assertion(DBG_ASSERTION(mixed_pop.size()));
//
//        //Rank the population according to Pareto fronts
//        front_t fronts;
//        _rank_crowd(mixed_pop, fronts);
//
//        //Add Pareto layers to the new population until the current layer no
//        //longer fits
//        new_pop.clear();
//        size_t front_index = 0;
//        while(fronts[front_index].size() + new_pop.size() < bin_size){
//            new_pop.insert(new_pop.end(), fronts[front_index].begin(),
//            		fronts[front_index].end());
//            ++front_index;
//        }
//
//        // sort the last layer
//        size_t size_remaining = bin_size - new_pop.size();
//        if (size_remaining > 0){
//            dbg::assertion(DBG_ASSERTION(front_index < fronts.size()));
//            std::sort(fronts[front_index].begin(), fronts[front_index].end(),
//            		crowd::compare_crowd());
//            for (size_t k = 0; k < size_remaining; ++k){
//                new_pop.push_back(fronts[front_index][k]);
//            }
//        }
//        dbg::assertion(DBG_ASSERTION(new_pop.size() == bin_size));
//    }
//
//    // --- rank & crowd ---
//
//    /**
//     * Ranks and crowds a population.
//     *
//     * Takes a population and divides it based on objectives.
//     *
//     * @param pop    The population to be ranked.
//     * @param fronts The resulting Pareto fronts will be stored here.
//     */
//    void _rank_crowd(pop_t& pop, front_t& fronts)
//    {
//        dbg::trace trace("ea", DBG_HERE);
//        //Execute ranking based on dominance
//        std::vector<size_t> ranks;
//        dom_sort_f()(pop, fronts, non_dom_f(), ranks);
//
//        //Why are we assigning a crowd score to every individual?
//        parallel::p_for(parallel::range_t(0, fronts.size()),
//        		crowd::assign_crowd<indiv_t >(fronts));
//    }
//
//
//    /**
//     * For the specified category and array, copies the category score to the
//     * obj_index (usually 1).
//     */
//    void _cat_to_obj(array_t& array, size_t category){
//        dbg::trace trace("ea", DBG_HERE);
//        dbg::check_bounds(dbg::error, 0, category, array.size(), DBG_HERE);
//        for(size_t i=0; i<array[category].size(); ++i){
//            array[category][i]->fit().set_obj(obj_index,
//            		array[category][i]->fit().getBinFitness(category));
//        }
//    }
//
//    /**
//     * Copies the stored diversity back to the relevant objective.
//     *
//     * Does nothing when DIV is not defined
//     */
//    void _div_to_obj(array_t& array, size_t category){
//        dbg::trace trace("ea", DBG_HERE);
//#if defined(DIV)
//        for(size_t i=0; i<bin_size; ++i){
//            size_t div_index = _array[category][i]->fit().objs().size() - 1;
//            array[category][i]->fit().set_obj(div_index,
//            		array[category][i]->fit().getBinDiversity(category));
//        }
//#endif
//    }
//
//    /**
//     * Copies the calculated diversity to the individuals diversity array.
//     *
//     * Does nothing when DIV is not defined
//     */
//    void _obj_to_div(array_t& array, size_t category){
//        dbg::trace trace("ea", DBG_HERE);
//#if defined(DIV)
//        for(size_t j = 0; j < array[category].size(); ++j){
//            array[category][j]->fit().initDiv();
//            size_t div_index = array[category][j]->fit().objs().size() - 1;
//            array[category][j]->fit().setDiv(category,
//            		array[category][j]->fit().obj(div_index));
//        }
//#endif
//    }
//
//    //Debug functions
//#ifdef DBG_ENABLED
//    array_t _debug_array;
//
//    enum array_type{
//        current_array,
//        debug_array
//    };
//
//    /**
//     * Prints the fitness and closest distance values for each position in the
//     * archive.
//     *
//     * Note: requires the _debug_array to be set, otherwise it will throw a
//     * segmentation fault
//     */
//    void _print_archive(){
//        dbg::trace trace("ea", DBG_HERE);
//        std::cout << "Old archive:" << std::endl;
//        _print_array(current_array);
//        std::cout << "New archive:" << std::endl;
//        _print_array(debug_array);
//        _init_debug_array();
//    }
//
//    /**
//     * Print the debug array.
//     */
//    void _print_array(array_type type){
//    	using namespace karma;
//        for(size_t i=0; i<nr_of_bins; ++i){
//            _cat_to_obj(_array, i);
//            _div_to_obj(_array, i);
//            _cat_to_obj(_debug_array, i);
//            _div_to_obj(_debug_array, i);
//
//            pop_t temp_current = _array[i];
//            pop_t temp_debug = _debug_array[i];
//
//            compare::sort(temp_current, compare::pareto_objs().descending());
//            compare::sort(temp_debug, compare::pareto_objs().descending());
//
//            for(size_t j=0; j<bin_size; ++j){
//                std::cout << "(";
//                for(size_t k=0; k<temp_current[j]->fit().objs().size(); ++k){
//                	float obj_old = temp_debug[j]->fit().obj(k);
//                	float obj_new = temp_current[j]->fit().obj(k);
//                    bool better = obj_old < obj_new;
//                    bool worse = obj_old > obj_new;
//                    if(better) std::cout << COL_GREEN;
//                    if(worse) std::cout << COL_MAGENTA;
//                    float value = temp_current[j]->fit().obj(k);;
//                    if(type == debug_array) value = temp_debug[j]->fit().obj(k);
//                    std::cout << format(
//                    		left_align(5, '0')[maxwidth(5)[double_]],
//                    		value);
//                    std::cout << END_COLOR;
//                    if(k+1 != _array[i][j]->fit().objs().size()) std::cout << ":";
//                }
//                std::cout << ") ";
//            }
//            std::cout << std::endl;
//        }
//    }
//
//    /**
//     * Sets the debug array.
//     *
//     * Required for debugging at the end of the random pop and load population
//     * functions.
//     */
//    void _init_debug_array(){
//        dbg::trace trace("ea", DBG_HERE);
//        for(size_t i=0; i<nr_of_bins; ++i){
//            _debug_array[i].resize(bin_size);
//            for(size_t j=0; j<bin_size; ++j){
//                _debug_array[i][j] = indiv_t(new crowd_t(*_array[i][j]));
//            }
//        }
//    }
//
//#endif //DBG_ENABLED
};
}
}

// Undefine everything
#undef DBOM
#undef DBOW
#undef DBOE

#endif /* MODULES_CMOEA_CMOEA_NSGA2_MPI_HPP_ */
