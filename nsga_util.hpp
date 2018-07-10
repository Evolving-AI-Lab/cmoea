/*
 * nsga_util.hpp
 *
 *  Created on: Mar 10, 2017
 *      Author: Joost Huizinga
 */

#ifndef MODULES_CMOEA_NSGA_UTIL_HPP_
#define MODULES_CMOEA_NSGA_UTIL_HPP_

#include <sferes/dbg/dbg.hpp>
#include <sferes/ea/dom_sort_basic.hpp>
#include <sferes/ea/common.hpp>
#include <sferes/ea/crowd.hpp>

namespace sferes {

//// Probabilistic domination sort
SFERES_CLASS(prob_dom_f){
public:
    template<typename Indiv>
    inline bool operator() (const Indiv &ind, const std::vector<Indiv>&pop) const
    {
        BOOST_FOREACH(Indiv i, pop){
            if (dominate_flag(i, ind) == 1)
                return false;
        }
        return true;
    }

    template<typename I1, typename I2>
    inline int dominate_flag(const I1& i1, const I2& i2) const
    {
        dbg::assertion(DBG_ASSERTION(i1->fit().objs().size()));
        dbg::assertion(DBG_ASSERTION(i2->fit().objs().size()));
        dbg::assertion(DBG_ASSERTION(i1->fit().objs().size() == i2->fit().objs().size()));

        size_t nb_objs = i1->fit().objs().size();

        dbg::out(dbg::info, "prob_dom") << "Objectives: " << nb_objs << std::endl;
        bool flag1 = false, flag2 = false;
        for (size_t i = 0; i < nb_objs; ++i)
        {
            dbg::assertion(DBG_ASSERTION(i < Params::obj_pressure_size()));
            float pressure = Params::obj_pressure(i);;

            dbg::out(dbg::info, "prob_dom") << "Objective: " << i << " pressure: " << pressure << " i1 value: " << i1->fit().obj(i) << " i2 value: " << i2->fit().obj(i) << std::endl;
            if (misc::rand<float>() > pressure)
                continue;
            float fi1 = i1->fit().obj(i);
            float fi2 = i2->fit().obj(i);
            if (fi1 > fi2) {
                flag1 = true;
            } else if (fi2 > fi1){
                flag2 = true;
            }
        }

        if (flag1 && !flag2){
            return 1;
        } else if (!flag1 && flag2){
            return -1;
        } else {
            return 0;
        }
    }
};


/**
 * Ranks and crowds a population.
 *
 * Takes a population and divides it based on objectives.
 *
 * @param pop    The population to be ranked.
 * @param fronts The resulting Pareto fronts will be stored here.
 */
template<typename Indiv, typename Sort, typename Comp>
void _rank_crowd(std::vector<Indiv>& pop, std::vector<std::vector<Indiv> >& fronts){
    dbg::trace trace("ea", DBG_HERE);
    //Execute ranking based on dominance
    std::vector<size_t> ranks;
    Sort()(pop, fronts, Comp(), ranks);

    //Why are we assigning a crowd score to every individual?
    parallel::p_for(parallel::range_t(0, fronts.size()), ea::crowd::assign_crowd<Indiv >(fronts));
}


/**
 * Takes a mixed population, sorts it according to Pareto dominance, and generates a new population
 * depending on the bin size.
 *
 * @Param mixed_pop The mixed population from which to select.
 *                  The mixed population must be larger than the bin_size for selection to occur.
 * @Param new_pop   Output parameter. After execution, should contain a number of individuals
 *                  equal to the bin_size, selected based on Pareto dominance first, crowding second.
 */
template<typename Indiv, typename Sort, typename Comp>
struct fill_dom_sort_f{
	typedef typename std::vector<Indiv> pop_t;
	typedef typename std::vector<pop_t> front_t;
    inline void operator() (pop_t& mixed_pop, pop_t& new_pop, size_t desired_size) const{
    	dbg::trace trace("ea", DBG_HERE);
    	dbg::assertion(DBG_ASSERTION(mixed_pop.size()));


    	//Rank the population according to Pareto fronts
    	dbg::out(dbg::info, "nsga") << "Ranking population" << std::endl;
    	front_t fronts;
    	_rank_crowd<Indiv, Sort, Comp>(mixed_pop, fronts);

    	//Add Pareto layers to the new population until the current layer no longer fits
    	new_pop.clear();
    	size_t front_index = 0;
    	while(fronts[front_index].size() + new_pop.size() < desired_size){
    		dbg::out(dbg::info, "nsga") <<
    				"Adding front: " << front_index <<
					" of size: " << fronts[front_index].size() << std::endl;
    		new_pop.insert(new_pop.end(), fronts[front_index].begin(), fronts[front_index].end());
    		++front_index;
    	}
    	dbg::out(dbg::info, "nsga") << "All full fronts added" << std::endl;

    	// sort the last layer
    	size_t size_remaining = desired_size - new_pop.size();
    	if (size_remaining > 0){
    		dbg::out(dbg::info, "nsga") << "Adding partial front" << std::endl;
    		dbg::assertion(DBG_ASSERTION(front_index < fronts.size()));
    		std::sort(fronts[front_index].begin(), fronts[front_index].end(), ea::crowd::compare_crowd());
    		for (size_t k = 0; k < size_remaining; ++k){
    			dbg::out(dbg::info, "nsga") << "Added indiv: " << k << std::endl;
    			new_pop.push_back(fronts[front_index][k]);
    		}
    	}
    }
};

template<typename Indiv, typename Sort, typename Comp>
void dom_sort(std::vector<Indiv>& mixed_pop, std::vector<Indiv>& new_pop, size_t desired_size){
	dbg::trace trace("ea", DBG_HERE);
	dbg::assertion(DBG_ASSERTION(mixed_pop.size()));
	typedef typename std::vector<Indiv> pop_t;
	typedef typename std::vector<pop_t> front_t;

	//Rank the population according to Pareto fronts
	front_t fronts;
	_rank_crowd<Indiv, Sort, Comp>(mixed_pop, fronts);

	//Add Pareto layers to the new population until the current layer no longer fits
	new_pop.clear();
	size_t front_index = 0;
	while(fronts[front_index].size() + new_pop.size() < desired_size){
		new_pop.insert(new_pop.end(), fronts[front_index].begin(), fronts[front_index].end());
		++front_index;
	}

	// sort the last layer
	size_t size_remaining = desired_size - new_pop.size();
	if (size_remaining > 0){
		dbg::assertion(DBG_ASSERTION(front_index < fronts.size()));
		std::sort(fronts[front_index].begin(), fronts[front_index].end(), ea::crowd::compare_crowd());
		for (size_t k = 0; k < size_remaining; ++k){
			new_pop.push_back(fronts[front_index][k]);
		}
	}
}

/**
 * Selects a random individual from the supplied population.
 */
template<typename Indiv>
Indiv selection(const std::vector<Indiv>& pop){
    dbg::trace trace("ea", DBG_HERE);
    dbg::assertion(DBG_ASSERTION(pop.size() > 0));
    // Random requires a population size greater than 1, so we need a check
    //if(pop.size() == 1) return pop[0];
    int x1 = misc::rand< int > (0, pop.size());
    dbg::check_bounds(dbg::error, 0, x1, pop.size(), DBG_HERE);
    return pop[x1];
}


template<typename pop1_t, typename pop2_t>
void convert_pop(const pop1_t& pop1, pop2_t& pop2){
    dbg::trace trace("ea", DBG_HERE);
    pop2.resize(pop1.size());
    for (size_t i = 0; i < pop1.size(); ++i){
        pop2[i] = pop1[i];
    }
}

/**
 * Converts a population from regular individuals to crowd individuals.
 */
template<typename crowd_t, typename pop1_t, typename pop2_t>
void convert_pop_to_crowd(const pop1_t& pop1, pop2_t& pop2){
    dbg::trace trace("ea", DBG_HERE);
    pop2.resize(pop1.size());
    for (size_t i = 0; i < pop1.size(); ++i){
        pop2[i] = boost::shared_ptr<crowd_t>(new crowd_t(*pop1[i]));
    }
}

} // namespace sferes

#endif /* MODULES_CMOEA_NSGA_UTIL_HPP_ */
