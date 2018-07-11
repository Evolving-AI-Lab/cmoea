/*
 * map_elite_innovation_nsga2.hpp
 *
 *  Created on: Mar 12, 2015
 *      Author: Joost Huizinga
 */

/**
 *
 * Using CMOEA requires the following parameters to be set
    struct cmoea_nsga
    {
        typedef sferes::ea::_dom_sort_basic::non_dominated_f non_dom_f;
    };

    struct cmoea
    {
        static const size_t bin_size = 10;
        static const size_t nb_of_bins = 63;
        static const size_t obj_index = 0;
    };
 */

#ifndef MODULES_CMOEA_CMOEA_NSGA2_HPP_
#define MODULES_CMOEA_CMOEA_NSGA2_HPP_

// Standard includes
#include <algorithm>
#include <limits>

// Boost includes
#include <boost/foreach.hpp>
#include <boost/multi_array.hpp>
#include <boost/array.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/include/for_each.hpp>

// Sferes includes
#include <sferes/stc.hpp>
#include <sferes/ea/ea.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/dbg/dbg.hpp>
#include <sferes/ea/dom_sort_basic.hpp>
#include <sferes/ea/common.hpp>
#include <sferes/ea/crowd.hpp>

// Module includes
#include <modules/datatools/formatting.hpp>
#include <modules/datatools/common_compare.hpp>
#include <modules/debugext/dbgext.hpp>

// Local includes
#include "cmoea_base.hpp"
#include "nsga_util.hpp"

// Debug defines
#define DBOE dbg::out(dbg::info, "ea")

namespace sferes
{
namespace ea
{

// Main class
SFERES_EA(CmoeaNsga2, CmoeaBase){
public:
    //Params

	// Type definitions
	typedef CmoeaNsga2<Phen, Eval, Stat, FitModifier, Params, Exact> this_t;
	typedef CmoeaBase<Phen, Eval, Stat, FitModifier, Params, Exact> parent_t;
    typedef typename parent_t::phen_t phen_t;
    typedef typename parent_t::phen_ptr_t phen_ptr_t;
    typedef typename parent_t::crowd_t crowd_t;
    typedef typename parent_t::indiv_t indiv_t;
    typedef typename parent_t::pop_t pop_t;
    typedef typename parent_t::front_t front_t;
    typedef typename parent_t::array_t array_t;
    SFERES_EA_FRIEND(CmoeaNsga2);

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

    //The type of non dominated comparator to use
    //Currently available types are:
    // - sferes::ea::_dom_sort_basic::non_dominated_f
    //   (defined in sferes/ea/dom_sort_basic.hpp)
    //   Regular comparisons based on dominance
    // - sferes::ea::innov_pnsga::prob_dom_f<Params>
    //   Comparisons based on probabilistic sorting, where some objectives can
    //   be stronger than others.
    typedef typename Params::cmoea_nsga::non_dom_f non_dom_f;

    //The index used to temporarily store the category
    //This index should hold a dummy value, as it will be overwritten
    //constantly. The default would be 0.
    static const size_t obj_index = Params::cmoea::obj_index;

    //The number of objectives (bins) used in the map
    size_t nr_of_bins;

    //Not actually the size of the standing population
    //(which is bin_size*nr_of_bins)
    //but the number of individuals that are generated each epoch.
    //Because individuals are always produced in pairs,
    //pop_size has to be divisible by 2.
    static const size_t indiv_per_gen = Params::pop::select_size;

    CmoeaNsga2()
    {
        dbg::trace trace("ea", DBG_HERE);
        //dbg::compile_assertion<pop_size%2 == 0>
        //("Population size has to be divisible by 2.");
        dbg::out(dbg::info, "ea") << "Objectives: " << nr_of_bins << std::endl;
        nr_of_bins = Params::cmoea::nb_of_bins;
        this->_array.resize(nr_of_bins);
    }


    void epoch()
    {
        dbg::trace trace("ea", DBG_HERE);

        // We are creating and selecting a number of individuals equal to the
        // population size. A simpler variant would only select and mutate one
        // individual
        pop_t ptmp;
        for (size_t i = 0; i < (indiv_per_gen/2); ++i)
        {
        	DBOE << "Creating individual: " << i <<std::endl;
            indiv_t p1 = this->_selection(this->_array);
            indiv_t p2 = this->_selection(this->_array);
            indiv_t i1, i2;

            p1->cross(p2, i1, i2);
            DBOE << "Mutating i1 " << i1 << std::endl;
            i1->mutate();
            DBOE << "Mutating i1 " << i1 << " success " << std::endl;

            DBOE << "Mutating i2 " << i2 << std::endl;
            i2->mutate();
            DBOE << "Mutating i2 " << i2 << " success" << std::endl;
            ptmp.push_back(i1);
            ptmp.push_back(i2);
        }
        this->_eval.eval(ptmp, 0, ptmp.size(), this->_fit_proto);
        stc::exact(this)->add_to_archive(ptmp);

        // Write the entire population to pop, so it can be dumped as a
        // checkpoint
        this->_convert_pop(this->_array, this->_pop);
    }
//
//    const array_t& archive() const { return this->_array; }

    /**
     * Adds the new `population' (vector of individuals) to the archive by
     * adding every individual to every bin, and then running NSGA 2 (or pNSGA)
     * selection on every bin.
     */
    void add_to_archive(pop_t& pop){
        dbg::trace trace("ea", DBG_HERE);
        //Add everyone to the archive
        for (size_t i = 0; i < nr_of_bins; ++i){
        	DBOE << "****** Processing category " << i << " ******" <<std::endl;
            pop_t new_bin;
            for(size_t j = 0; j < pop.size(); ++j){
            	this->_array[i].push_back(pop[j]);
            }
            this->apply_modifier(this->_array[i]);
            _cat_to_obj(this->_array, i);
            _fill_nondominated_sort(this->_array[i], new_bin);
            this->_array[i] = new_bin;
        }
    }

protected:

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
        dbg::assertion(DBG_ASSERTION(mixed_pop.size()));

        //Rank the population according to Pareto fronts
        front_t fronts;
        _rank_crowd(mixed_pop, fronts);

        //Add Pareto layers to the new population until the current layer no
        // longer fits
        new_pop.clear();
        size_t front_index = 0;
        while(fronts[front_index].size() + new_pop.size() < this->bin_size){
            new_pop.insert(new_pop.end(), fronts[front_index].begin(), fronts[front_index].end());
            ++front_index;
        }

        // sort the last layer
        size_t size_remaining = this->bin_size - new_pop.size();
        if (size_remaining > 0){
            dbg::assertion(DBG_ASSERTION(front_index < fronts.size()));
            std::sort(fronts[front_index].begin(), fronts[front_index].end(), crowd::compare_crowd());
            for (size_t k = 0; k < size_remaining; ++k){
                new_pop.push_back(fronts[front_index][k]);
            }
        }
        dbg::assertion(DBG_ASSERTION(new_pop.size() == this->bin_size));
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
        parallel::p_for(parallel::range_t(0, fronts.size()), crowd::assign_crowd<indiv_t >(fronts));

        //Why are we sorting the population?
//        for (size_t i = 0; i < ranks.size(); ++i){
//            pop[i]->set_rank(ranks[i]);
//        }
//        parallel::sort(pop.begin(), pop.end(), crowd::compare_ranks());;
    }


    /**
     * For the specified category and array, copies the category score to the
     * obj_index (usually 1).
     */
    void _cat_to_obj(array_t& array, size_t category){
        dbg::trace trace("ea", DBG_HERE);
        dbg::check_bounds(dbg::error, 0, category, array.size(), DBG_HERE);
        for(size_t i=0; i<array[category].size(); ++i){
            array[category][i]->fit().set_obj(obj_index, array[category][i]->fit().getBinFitness(category));
        }
    }

    /**
     * Copies the stored diversity back to the relevant objective.
     *
     * Does nothing when DIV is not defined
     */
    void _div_to_obj(array_t& array, size_t category){
        dbg::trace trace("ea", DBG_HERE);
#if defined(DIV)
        for(size_t i=0; i<this->bin_size; ++i){
            size_t div_index = this->_array[category][i]->fit().objs().size() - 1;
            array[category][i]->fit().set_obj(div_index, array[category][i]->fit().getBinDiversity(category));
        }
#endif
    }

    /**
     * Copies the calculated diversity to the individuals diversity array.
     *
     * Does nothing when DIV is not defined
     */
    void _obj_to_div(array_t& array, size_t category){
        dbg::trace trace("ea", DBG_HERE);
#if defined(DIV)
        for(size_t j = 0; j < array[category].size(); ++j){
            array[category][j]->fit().initBinDiversity();
            size_t div_index = array[category][j]->fit().objs().size() - 1;
            array[category][j]->fit().setBinDiversity(category, array[category][j]->fit().obj(div_index));
        }
#endif
    }
};
}
}

// Un-define all debug macros to avoid interactions
#undef DBOE

#endif /* MODULES_CMOEA_CMOEA_NSGA2_HPP_ */
