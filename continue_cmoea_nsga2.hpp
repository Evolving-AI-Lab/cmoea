/*
 * continue_cmoea_nsga2.hpp
 *
 *  Created on: Mar 12, 2015
 *      Author: Joost Huizinga
 */

#ifndef MODULES_CMOEA_CONTINUE_CMOEA_NSGA2_HPP_
#define MODULES_CMOEA_CONTINUE_CMOEA_NSGA2_HPP_

//Sferes includes
#include <sferes/dbg/dbg.hpp>
#include <sferes/ea/crowd.hpp>
#include <sferes/ea/ea.hpp>

//Boost includes
#include <boost/serialization/binary_object.hpp>
#include <boost/fusion/container.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/assign/list_of.hpp>

namespace sferes{
namespace ea{

template<typename EA, typename Params>
class ContinueCmoeaNsga2 : public EA{
public:

#ifdef  SFERES_XML_WRITE
    typedef boost::archive::xml_oarchive oa_t;
#else
    typedef boost::archive::binary_oarchive oa_t;
#endif


    void set_gen(size_t gen){
        this->_gen = gen;
    }

    void write_gen_file(const std::string& prefix){
        std::string fname = this->_res_dir + std::string("/") + prefix + boost::lexical_cast<std::string>(this->_gen);
        this->_write(fname, this->_stat);
    }

    template<typename LocalPhen>
    void init_parent_pop(std::vector<boost::shared_ptr<LocalPhen> > population){
        typename EA::pop_t converted_pop;
        this->_convert_pop(population, converted_pop);

        dbg::assertion(DBG_ASSERTION(population.size() == converted_pop.size()));
        dbg::assertion(DBG_ASSERTION(converted_pop.size() == this-> _array.size()*this->bin_size));

        dbg::out(dbg::info, "continue") << "Adding: " << converted_pop.size()
                << " to archive: " << this-> _array.size()
                << " by " << this->bin_size << std::endl;

        //Add everyone to the archive in the appropriate place
        size_t pop_index = 0;
        for(size_t i=0; i<this->_array.size(); ++i){
            for(size_t j=0; j<this->bin_size; ++j){
                this->_array[i].push_back(converted_pop[pop_index]);
                ++pop_index;
            }
        }

        DBG_CONDITIONAL(dbg::info, "archive", this->_init_debug_array());
    }

protected:


    /**
     * Writes all stats to the archive with the supplied name.
     *
     * If BINARYARCHIVE is defines, also writes a final string,
     * used to assert that the file has been written properly.
     *
     * @param fname The name if the output file.
     * @param stat  The boost fusion vector of statistics that should be written.
     */
    template<typename ConvertStatType>
    void _write(const std::string& fname, ConvertStatType& stat) const
    {
      dbg::trace trace("ea", DBG_HERE);
      std::ofstream ofs(fname.c_str());
      oa_t oa(ofs);
      boost::fusion::for_each(stat, WriteStat_f<oa_t>(oa));

#if !defined(SFERES_XML_WRITE)
      //If our archive is binary, explicitly set these characters at the end of the file
      //so we can check (or at least be reasonably certain) that the archive was written
      //successfully and entirely
      char end[25] = "\n</boost_serialization>\n";
      boost::serialization::binary_object object(end, 25);
      oa << object;
#endif

      std::cout << fname << " written" << std::endl;
    }
};

}
}

#endif /* MODULES_MAPELITE_CONTINUE_MAPELITE_INNOV_NSGA2_HPP_ */
