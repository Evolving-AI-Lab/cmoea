/*
 * cmoea_base.hpp
 *
 *  Created on: Jun 8, 2017
 *      Author: joost
 */

#ifndef MODULES_CMOEA_CMOEA_BASE_HPP_
#define MODULES_CMOEA_CMOEA_BASE_HPP_

// Boost includes
#include <boost/foreach.hpp>

// Sferes includes
#include <sferes/stc.hpp>
#include <sferes/ea/ea.hpp>
#include <sferes/dbg/dbg.hpp>

// Local includes
#include "cmoea_util.hpp"

// Debug defines
#define DBOE dbg::out(dbg::info, "ea")

namespace sferes
{
namespace ea
{

// Main class
SFERES_EA(CmoeaBase, Ea){
public:
	// Type definitions
	typedef Ea<Phen, Eval, Stat, FitModifier, Params, Exact> parent_t;
    typedef CmoeaBase<Phen, Eval, Stat, FitModifier, Params, Exact> this_t;
    typedef Phen phen_t;
    typedef boost::shared_ptr<phen_t> phen_ptr_t;
    typedef crowd::Indiv<phen_t> crowd_t;
    typedef boost::shared_ptr<crowd_t> indiv_t;
    typedef typename std::vector<indiv_t> pop_t;
    typedef typename std::vector<std::vector<indiv_t> > front_t;
    typedef std::vector<pop_t> array_t;
    SFERES_EA_FRIEND(CmoeaBase);

    // The number of initially created individuals.
    static const size_t init_size = Params::pop::init_size;

    // Max batch size to avoid memory issues
    static const size_t max_batch_size = Params::pop::max_batch_size;

    //The size of each bin
    static const size_t bin_size = Params::cmoea::bin_size;

    void random_pop()
    {
        dbg::trace trace("ea", DBG_HERE);
        //Create and evaluate the initial random population
        parallel::init();
        unsigned nr_of_batches = (init_size / max_batch_size);
        unsigned remainder = init_size % max_batch_size;
        if(remainder) ++nr_of_batches;
        DBOE << "Nr of batches: " << nr_of_batches << std::endl;
        for(unsigned j=0; j<nr_of_batches; ++j){
        	unsigned batch_size = max_batch_size;
        	if(j==(nr_of_batches-1) && remainder > 0){
        		batch_size = remainder;
        	}
        	DBOE << "Processing batch: " << j <<
        			" of size: " << batch_size << std::endl;
            pop_t pop;
            pop.resize(batch_size);
            int i = 0;
            BOOST_FOREACH(indiv_t& indiv, pop)
            {
            	DBOE << "Creating random individual: " << i++ << std::endl;
                indiv = indiv_t(new crowd_t());
                indiv->random();
            }
            DBOE << "Evaluating population" << std::endl;
            this->_eval.eval(pop, 0, pop.size(), this->_fit_proto);

            stc::exact(this)->apply_modifier(pop);
            stc::exact(this)->add_to_archive(pop);
        }
    }

    /**
     * Applies the modifier to the supplied population (vector of individuals).
     *
     * Note that this overwrites the this->_pop population.
     */
    void apply_modifier(pop_t pop){
        dbg::trace trace("ea", DBG_HERE);
        _convert_pop(pop, this->_pop);
        parent_t::apply_modifier();
    }

    const array_t& archive() const { return _array; }

protected:
    array_t _array;

    /**
     * Converts a population from array individuals to regular individuals.
     */
    void _convert_pop(const pop_t& pop1, std::vector<phen_ptr_t>& pop2){
    	dbg::trace trace("ea", DBG_HERE);
    	pop2.resize(pop1.size());
    	for (size_t i = 0; i < pop1.size(); ++i){
    		pop2[i] = pop1[i];
    	}
    }


    /**
     * Converts the entire array of individuals to regular individuals.
     */
    void _convert_pop(const array_t& array, std::vector<phen_ptr_t>& pop2){
    	dbg::trace trace("ea", DBG_HERE);
    	pop2.resize(array.size() * bin_size);
    	size_t k=0;
    	for (size_t i = 0; i < array.size(); ++i){
    		for (size_t j = 0; j < bin_size; ++j){
    			pop2[k++] = array[i][j];
    		}
    	}
    }

    /**
     * Converts a population from regular individuals to crowd individuals.
     */
    void _convert_pop(const std::vector<phen_ptr_t>& pop1, pop_t& pop2){
    	dbg::trace trace("ea", DBG_HERE);
    	pop2.resize(pop1.size());
    	for (size_t i = 0; i < pop1.size(); ++i){
    		pop2[i] = boost::shared_ptr<crowd_t>(new crowd_t(*pop1[i]));
    	}
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
     * Selects a random individual from the supplied archive
     */
    indiv_t _selection(const array_t& archive){
        dbg::trace trace("ea", DBG_HERE);
        size_t category = misc::rand< size_t > (0, archive.size());
        dbg::check_bounds(dbg::error, 0, category, archive.size(), DBG_HERE);
        size_t individual_index = misc::rand< size_t > (0, archive[category].size());
        dbg::check_bounds(dbg::error, 0, individual_index, archive[category].size(), DBG_HERE);
        return archive[category][individual_index];
    }


    void _set_pop(const std::vector<boost::shared_ptr<Phen> >& population) {
        dbg::trace trace("ea", DBG_HERE);
        pop_t converted_pop;
        this->_convert_pop(population, converted_pop);

        dbg::out(dbg::info, "continue") << "Adding: " << converted_pop.size()
                << " to archive: " << this-> _array.size()
                << " by " << this->bin_size << std::endl;
        dbg::assertion(DBG_ASSERTION(population.size() == converted_pop.size()));
        dbg::assertion(DBG_ASSERTION(converted_pop.size() == this->_array.size()*this->bin_size));

        //Add everyone to the archive in the appropriate place
        size_t pop_index = 0;
        for(size_t i=0; i<this->_array.size(); ++i){
            for(size_t j=0; j<this->bin_size; ++j){
                this->_array[i].push_back(converted_pop[pop_index]);
                ++pop_index;
            }
        }
    }
};
}
}

// Un-define all debug macros to avoid interactions
#undef DBOE

#endif /* MODULES_CMOEA_CMOEA_BASE_HPP_ */
