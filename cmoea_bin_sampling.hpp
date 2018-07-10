/*
 * cmoea_bin_sampling.hpp
 *
 *  Created on: Jun 1, 2017
 *      Author: Joost Huizinga
 */
#ifndef MODULES_CMOEA_CMOEA_BIN_SAMPLING_HPP_
#define MODULES_CMOEA_CMOEA_BIN_SAMPLING_HPP_

// Standard includes
#include <algorithm>
#include <limits>

// Boost includes
#include <boost/foreach.hpp>
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
#include <modules/misc/formatting.hpp>
#include <modules/misc/common_compare.hpp>
#include <modules/misc/clock.hpp>
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
SFERES_EA(CmoeaBinSampling, CmoeaBase){
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
    //size_t nr_of_bins;

    // The number of different tasks that need to be performed
    size_t nr_of_objs;

    double obj_change_rate;

    // The maximum number of bins instantiated at any one time
    static const size_t max_bins = Params::cmoea::nb_of_bins;

    //The size of each bin
    static const size_t bin_size = Params::cmoea::bin_size;

    // The number of individuals initially generated to fill the archive
    // If equal to the bin_size, every initially generated individual is added
    // to every bin of every category.
    // The init_size has to be greater than or equal to the bin_size
    static const size_t init_size = Params::pop::init_size;

    // Very large initial populations may cause CMOEA to run out of memory.
    // To avoid this, you can add the initial populations in init_batch batches
    // of init_size.
    static const size_t max_batch_size = Params::pop::max_batch_size;

    // The number of individuals created per generation
    static const size_t indiv_per_gen = Params::pop::select_size;

    typedef Phen phen_t;
    typedef boost::shared_ptr<Phen> phen_ptr_t;
    typedef crowd::Indiv<phen_t> crowd_t;
    typedef boost::shared_ptr<crowd_t> indiv_t;
    typedef std::vector<indiv_t> pop_t;
    typedef std::vector<pop_t> front_t;
    typedef std::vector<pop_t> array_t;
    // While supposed to be a vector of booleans, the data() function, called
    // by boost mpi, is not provided for the vector<bool> specialization.
    typedef std::vector<char> bitset_t;
    typedef CmoeaBinSampling<Phen, Eval, Stat, FitModifier, Params,
    		Exact> this_t;

    // The exact type of non-dominated sort function to use
    typedef fill_dom_sort_f<indiv_t, dom_sort_f, non_dom_f> sort_f;

    SFERES_EA_FRIEND(CmoeaBinSampling);

    CmoeaBinSampling()
    {
        dbg::trace trace("ea", DBG_HERE);
        nr_of_objs = Params::cmoea::nr_of_objs;
        DBOE << "Objectives: " << nr_of_objs << std::endl;
        obj_change_rate = Params::cmoea::obj_change_rate;

        // Make sure settings are sane
        dbg::assertion(DBG_ASSERTION(init_size >= bin_size));

        // Set the size of the relevant vectors
        this->_array.resize(max_bins);
        _obj_combs.resize(max_bins);

        // Add the combination of all objectives
        bitset_t _all_objs(nr_of_objs, true);
        _obj_combs[0] = _all_objs;

        // Add each objective separately
        for(unsigned i=0; i<nr_of_objs; ++i){
        	bitset_t _single_obj(nr_of_objs, false);
        	_single_obj[i] = true;
        	_obj_combs[i+1] = _single_obj;
        }

        // Fill the rest with random bins
        for(unsigned i=nr_of_objs+1; i<_obj_combs.size(); ++i){
        	bitset_t _random_objs(nr_of_objs, false);
        	for(unsigned j=0; j<nr_of_objs; ++j){
        		_random_objs[j] = misc::flip_coin();
        	}
        	_obj_combs[i] = _random_objs;
        }

        // Create clocks
        _clocks["repro"] = Clock();
        _clocks["eval"] = Clock();
        _clocks["select"] = Clock();
    }


    void epoch()
    {
        dbg::trace trace("ea", DBG_HERE);
        DBOE << "Epoch start" << std::endl;
        // We are creating and selecting a number of individuals equal to the
        // population size. A simpler variant would only select and mutate one
        // individual
        _clocks["repro"].resetAndStart();
        pop_t ptmp;
        for (size_t i = 0; i < (indiv_per_gen/2); ++i)
        {
        	DBOE << "Creating indivs: " << i*2 << " and " << i*2 + 1 <<std::endl;
            indiv_t p1 = this->_selection(this->_array);
            indiv_t p2 = this->_selection(this->_array);
            indiv_t i1, i2;
            p1->cross(p2, i1, i2);
            i1->mutate();
            i2->mutate();
            ptmp.push_back(i1);
            ptmp.push_back(i2);
        }
        _clocks["repro"].stop();
        _clocks["eval"].resetAndStart();
        DBOE << "Evaluating population" << std::endl;
        this->_eval.eval(ptmp, 0, ptmp.size(), this->_fit_proto);
        _clocks["eval"].stop();
        _clocks["select"].resetAndStart();
        DBOE << "Performing selection" << std::endl;
        stc::exact(this)->add_to_archive(ptmp);
        _clocks["select"].stop();

        // Write the entire population to pop, so it can be dumped as a
        // checkpoint
        DBOE << "Converting population" << std::endl;
        this->_convert_pop(this->_array, this->_pop);

        // Change objectives
        DBOE << "Changing objectives" << std::endl;
        _change_objs();
        DBOE << "Epoch done" << std::endl;
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
        for (size_t i = 0; i < max_bins; ++i){
        	DBOE << "****** Processing bin " << i << " ******" <<std::endl;
#if defined(DBG_ENABLED)
        	std::string bits;
        	for(size_t j=0; j<_obj_combs[i].size(); ++j){
        		if(_obj_combs[i][j]){
        			bits += "1";
        		} else {
        			bits += "0";
        		}
        	}
        	DBOE << "Objectives: " << bits << std::endl;
#endif
            pop_t new_bin;
            for(size_t j = 0; j < pop.size(); ++j){
            	this->_array[i].push_back(pop[j]);
            }
            this->apply_modifier(this->_array[i]);
            _cat_to_obj(this->_array, i);
        	sort_f()(this->_array[i], new_bin, bin_size);
        	this->_array[i] = new_bin;
        }
    }

    std::map<std::string, Clock> getClocks() const{
    	return _clocks;
    }

protected:
    std::vector<bitset_t> _obj_combs;
    std::map<std::string, Clock> _clocks;


    void _change_objs(){
        for(unsigned i=nr_of_objs+1; i<_obj_combs.size(); ++i){
        	if(misc::rand<double>() < obj_change_rate){
        		DBOE << "Objective changed: " << i << std::endl;
        		bitset_t _random_objs(nr_of_objs, false);
        		for(unsigned j=0; j<nr_of_objs; ++j){
        			_random_objs[j] = misc::flip_coin();
        		}
        		_obj_combs[i] = _random_objs;
        	}
        }
    }


    /**
     * For the specified category and array, copies the category score to the
     * obj_index (usually 1).
     */
    void _cat_to_obj(array_t& array, size_t index){
        dbg::trace trace("ea", DBG_HERE);
        dbg::check_bounds(dbg::error, 0, index, array.size(), DBG_HERE);
        for(size_t i=0; i<array[index].size(); ++i){
        	double fit = 1;
        	for(size_t obj_i=0; obj_i<nr_of_objs; ++obj_i){
        		if(_obj_combs[index][obj_i]){
        			fit *= array[index][i]->fit().getCmoeaObj(obj_i);
        		}
        	}
        	DBOE << "Indiv: " << i << " fit: " << fit << std::endl;
            array[index][i]->fit().set_obj(obj_index, fit);
        }
    }
};
}
}

// Un-define all debug macros to avoid interactions
#undef DBOE

#endif /* MODULES_CMOEA_CMOEA_BIN_SAMPLING_HPP_ */
