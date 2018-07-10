/*
 * mpi_util_test.cpp
 *
 *  Created on: Mar 24, 2017
 *      Author: Joost Huizinga
 *
 * Because this tests mpi, the test should be run with:
 *  mpirun -np 2 build/modules/cmoea/mpi_util_test
 *
 * Because the boost test suite is the first to parse parameters, to enable
 * debug output, you should call this test case as follows:
 *  mpirun -np 2 build/modules/cmoea/mpi_util_test -- --verbose mpi
 */

// Defines
#ifndef STATIC_BOOST
#define BOOST_TEST_DYN_LINK
#endif
#define BOOST_TEST_MODULE cmoea_mpi_util

// Always enable debugging for our test cases
#define DBG_ENABLED
#undef NDEBUG

// Boost includes
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

// Module includes
#include <modules/continue/global_options.hpp>

// Local includes
#include "mpi_util.hpp"

// Local debug macro
#define DBO dbg::out(dbg::info, "mpi") << "["<< comm->rank() << "] "

// Name spaces
using namespace sferes;
using namespace sferes::ea;

/**
* Make available program's arguments to all tests, receiving this fixture.
*
* These arguments ensure that debugging statements can be enabled, which is
* useful for when the test fails.
*/
struct ArgsFixture {
   ArgsFixture(): argc(boost::unit_test::framework::master_test_suite().argc),
           argv(boost::unit_test::framework::master_test_suite().argv){}
   int argc;
   char **argv;
};

static const int data_size = 4;

BOOST_FIXTURE_TEST_CASE(mpi_util_test, ArgsFixture){
	std::cout << "Test mpi_util_test started." << std::endl;
    sferes::options::parse_and_init(argc, argv, false);

    // Initialize MPI
    com_t comm = 0;
	env_t env = 0;
	init(comm, env);
	BOOST_CHECK(comm);
	BOOST_CHECK(env);

	// Received
	std::vector<bool> packets_received(comm->size()*data_size, false);

	// Define some data unique for each MPI instance
	std::vector<int> data;
	for (int i=0; i<data_size; ++i){
		data.push_back(i + comm->rank()*data_size);
	}

	// Test global vector exchange
	std::vector<int> recv_data;
	g_exchange_v(comm, data, recv_data);

	for (int i=0; i<recv_data.size(); ++i){
		DBO << " got data " << recv_data[i] << std::endl;
		packets_received[recv_data[i]] = true;
	}

	// Make sure all processes received all packets
	std::string recv_as_string;
	for (int i=0; i<packets_received.size(); ++i){
		BOOST_TEST(packets_received[i]);
		if(packets_received[i]){
			recv_as_string += "1";
		} else {
			recv_as_string += "0";
		}
	}
	DBO << "received: " << recv_as_string << std::endl;

	MPI_Finalize();
}

#undef DBO
