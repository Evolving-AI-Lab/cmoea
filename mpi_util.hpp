/*
 * mpi_util.hpp
 *
 *  Created on: Mar 10, 2017
 *      Author: Joost Huizinga
 */

#ifndef MODULES_CMOEA_MPI_UTIL_HPP_
#define MODULES_CMOEA_MPI_UTIL_HPP_

// Boost includes
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/detail/point_to_point.hpp>

// Defines
#define DBO dbg::out(dbg::info, "mpi") << "["<< comm->rank() << "] "

namespace sferes
{
namespace ea
{

typedef typename boost::mpi::environment mpi_env_t;
typedef typename boost::mpi::communicator mpi_com_t;
typedef boost::shared_ptr<mpi_com_t> com_t;
typedef boost::shared_ptr<mpi_env_t> env_t;
typedef std::vector<MPI_Request*> rv_t;
typedef std::vector<unsigned long*> sv_t;
typedef typename boost::mpi::packed_oarchive mpi_oa_t;
typedef typename boost::mpi::packed_iarchive mpi_ia_t;


static const size_t max_send_size = 500;

inline void init(com_t& comm, env_t& env){
    // Initialize MPI
	static char* argv[] = {(char*)"sferes2", 0x0};
	char** argv2 = (char**) malloc(sizeof(char*) * 2);
	int argc = 1;
	argv2[0] = argv[0];
	argv2[1] = argv[1];
	using namespace boost;
	dbg::out(dbg::info, "mpi") << "Initializing MPI..." << std::endl;
	env = env_t(new mpi_env_t(argc, argv2, true));
	dbg::out(dbg::info, "mpi") << "MPI initialized" << std::endl;
	comm = com_t(new mpi_com_t());
}

inline int get_half(const int& world_size){
    return (world_size/2) + 1;
}

inline int get_my_world_size(int rank, int world_size, int& source){
    int half;
    int next_source = 0;
    while(rank > 0){
        source = next_source;
        half = get_half(world_size);
        if(rank >= half){
            rank -= half;
            world_size = (world_size-1)/2;
            next_source += half;
        } else {
            rank -= 1;
            world_size = world_size/2;
            next_source += 1;
        }
    }
    return world_size;
}

/**
 * A non-blocking send, which will allocate memory for requests and sending
 * archive size. Needs to be matched with a wait_all to free all pointers.
 */
template<typename Ar_t> inline
void isend(com_t comm, Ar_t& oa, int target, int tag, rv_t& reqs, sv_t& sizes){
	DBO << "sending to rank: " << target << std::endl;
    int r;
	MPI_Request* req1 = new MPI_Request();
	MPI_Request* req2 = new MPI_Request();
    unsigned long* size_p = new unsigned long(oa.size());
    const void* data_p = const_cast<void*>(oa.address());
    r = MPI_Isend(size_p, 1, MPI_UNSIGNED_LONG, target, tag, *comm, req1);
    if (r != 0) boost::throw_exception(boost::mpi::exception("MPI_Isend", r));
    r = MPI_Isend(data_p, oa.size(), MPI_PACKED, target, tag, *comm, req2);
    if (r != 0) boost::throw_exception(boost::mpi::exception("MPI_Isend", r));
	reqs.push_back(req1);
	reqs.push_back(req2);
	sizes.push_back(size_p);
}


/**
 * Receive individuals from the indicated MPI instance.
 *
 * @param partner: The rank of the MPI instance to receive from.
 * @param pop (out): Reference to a population in which to store the
 *     received individuals.
 */
template<typename Ar_t>
void irecv(com_t comm, Ar_t& ia, int source, int tag){
	dbg::trace trace("ea", DBG_HERE);
    // Receive data from the root.
    MPI_Status stat;
    unsigned long count;
    int recv = 0;
    using namespace boost::mpi;

    DBO << "receiving from rank: " << source << std::endl;

    BOOST_MPI_CHECK_RESULT(MPI_Recv,
            (&count, 1,  MPI_UNSIGNED_LONG,
            source, tag, *comm, &stat));
    MPI_Get_count(&stat,  MPI_UNSIGNED_LONG, &recv);
    DBO  << " received size_t of size: " << recv <<
			" from " << stat.MPI_SOURCE <<
			" with tag " << stat.MPI_TAG << std::endl;

    // Prepare input buffer and receive the message
    //mpi_ia_t ia(*comm);
    ia.resize(count);
    BOOST_MPI_CHECK_RESULT(MPI_Recv,
            (ia.address(), ia.size(), MPI_PACKED,
            source, tag,
             *comm, &stat));
    MPI_Get_count(&stat, MPI_PACKED, &recv);
    DBO <<  " received archive of size: " << recv <<
			" from " << stat.MPI_SOURCE <<
			" with tag " << stat.MPI_TAG << std::endl;

    // De-serializing population
    //ia >> pop;
}

/**
 * Waits until all requests are completed, and then frees all allocated memory.
 */
inline void wait_all(rv_t& reqs, sv_t& sizes){
	std::vector<MPI_Request> requests(reqs.size());
	for(size_t i=0; i<reqs.size(); ++i){
		requests[i] = *(reqs[i]);
	}
	BOOST_MPI_CHECK_RESULT(MPI_Waitall, (requests.size(), &requests[0], MPI_STATUSES_IGNORE));
	for(size_t i=0; i<reqs.size(); ++i){
		delete reqs[i];
	}
	reqs.clear();
	for(size_t i=0; i<sizes.size(); ++i){
		delete sizes[i];
	}
	sizes.clear();
//	for(size_t i=0; i<_send_buffer.size(); ++i){
//		delete _send_buffer[i];
//	}
//	_send_buffer.clear();
}

template<typename Archive_t>
inline void broadcast_send(boost::shared_ptr<boost::mpi::communicator> comm, Archive_t& archive, int& my_world_size, int tag){
    using namespace boost::mpi;
    if(my_world_size > 1){
        std::vector<MPI_Request*> requests;
        std::vector<size_t*> sizes;
        //int num_requests = 0;
        int target_1 = comm->rank() + 1;
//        dbg::out(dbg::info, "mpi") << "Rank: " << comm->rank() << " sending to rank: " << target_1 << std::endl;
        isend(comm, archive, target_1, tag, requests, sizes);
//        const void* size_1 = &archive.size();
//        BOOST_MPI_CHECK_RESULT(MPI_Isend,
//                (const_cast<void*>(size_1), 1,
//                 get_mpi_datatype<size_t>(archive.size()),
//                 target_1, tag, *comm, &requests[0]));
//        BOOST_MPI_CHECK_RESULT(MPI_Isend,
//                (const_cast<void*>(archive.address()), archive.size(),
//                 MPI_PACKED,
//                 target_1, tag, *comm, &requests[1]));
        //num_requests = 2;


        //num_requests += detail::packed_archive_isend(*comm, target_1, tag, archive, &requests[num_requests], 2);
        if(my_world_size > 2){
            int target_2 = comm->rank() + get_half(my_world_size);
            isend(comm, archive, target_2, tag, requests, sizes);
//            dbg::out(dbg::info, "mpi") << "Rank: " << comm->rank() << " sending to rank: " << target_2 << std::endl;
//            const void* size_2 = &archive.size();
//            BOOST_MPI_CHECK_RESULT(MPI_Isend,
//                    (const_cast<void*>(size_2), 1,
//                     get_mpi_datatype<size_t>(archive.size()),
//                     target_2, tag, *comm, &requests[2]));
//            BOOST_MPI_CHECK_RESULT(MPI_Isend,
//                    (const_cast<void*>(archive.address()), archive.size(),
//                     MPI_PACKED,
//                     target_2, tag, *comm, &requests[3]));
//            num_requests = 4;
            //num_requests += detail::packed_archive_isend(*comm, target_2, tag, archive, &requests[num_requests], 2);
        }
        wait_all(requests, sizes);
        //BOOST_MPI_CHECK_RESULT(MPI_Waitall, (num_requests, &requests[0], MPI_STATUSES_IGNORE));
    }
}

template<typename Data_t>
void better_broadcast(boost::shared_ptr<boost::mpi::communicator> comm, Data_t& data){
    using namespace boost::mpi;
    int tag = environment::collectives_tag();
    int size = comm->size();

    if (comm->rank() == 0) {
        packed_oarchive oa(*comm);
        oa << data;
        dbg::out(dbg::info, "mpi") << "Master " << comm->rank()  << " send archive: " <<  oa.size() << std::endl;
        broadcast_send(comm, oa, size, tag);
    } else {
        int source=0;
        int my_world_size = get_my_world_size(comm->rank(), size, source);
        packed_iarchive ia(*comm);
        //broadcast(comm, ia, root);

        // Receive data from the root.
        MPI_Status stat;
        size_t count;
        int number_amount = 0;

        BOOST_MPI_CHECK_RESULT(MPI_Recv,
                (&count, 1, get_mpi_datatype<size_t>(count),
                 source, tag, *comm, &stat));
        MPI_Get_count(&stat, get_mpi_datatype<size_t>(count), &number_amount);
        dbg::out(dbg::info, "mpi") << "Worker " << comm->rank()  << " received size_t of size: " << number_amount << " from " << stat.MPI_SOURCE << " with tag " << stat.MPI_TAG << std::endl;

        // Prepare input buffer and receive the message
        ia.resize(count);
        BOOST_MPI_CHECK_RESULT(MPI_Recv,
                (ia.address(), ia.size(), MPI_PACKED,
                 source, tag,
                 *comm, &stat));
        MPI_Get_count(&stat, MPI_PACKED, &number_amount);
        dbg::out(dbg::info, "mpi") << "Worker " << comm->rank()  << " received archive of size: " << number_amount << " from " << stat.MPI_SOURCE << " with tag " << stat.MPI_TAG << std::endl;

        //detail::packed_archive_recv(*comm, source, tag, ia, status);
        dbg::out(dbg::info, "mpi") << "Worker " << comm->rank()  << " Archive received: " <<  ia.size() << std::endl;
        broadcast_send(comm, ia, my_world_size, tag);

        ia >> data;
        //for (int i = 0; i < n; ++i)
        //    ia >> values[i];
    }
}

template<typename oa_t, typename ia_t> inline
void l_ex(com_t comm, oa_t& a_send, ia_t& a_recv, int t, int tag, bool rev){
    std::vector<MPI_Request*> reqs;
    std::vector<size_t*> sizes;
    if(rev){
    	irecv(comm, a_recv, t, tag);
    	isend(comm, a_send, t, tag, reqs, sizes);
    } else {
    	isend(comm, a_send, t, tag, reqs, sizes);
    	irecv(comm, a_recv, t, tag);
    }
	wait_all(reqs, sizes);
}

template<typename Data_t> inline
void g_exchange(com_t comm, Data_t& to_send, std::vector<Data_t>& to_recv){
    using namespace boost::mpi;
    int tag = environment::collectives_tag();
    int size = comm->size();
    int rank = comm->rank();

    mpi_oa_t oa(*comm);
    oa << to_send;
    std::vector<mpi_ia_t*> ia;
    for(int i=0; i<(size-1); ++i){
    	ia.push_back(new mpi_ia_t(*comm));
    }
    //mpi_ia_t ia;

    int target;
    int k=0;
    bool rev;
    for(int i = 1; i<comm->size(); i*=2){
    	DBO << "offset: " << i << " out of " << comm->size() << std::endl;
    	if(((rank / i) % 2) == 0){
    		target = rank+i;
    		rev = false;
    	} else {
    		target = rank-i;
    		rev = true;
    	}
    	l_ex(comm, oa, *ia[k], target, tag, rev);
    	int prev_received = k;
    	++k;
    	for(int j=0; j<prev_received; ++j){
    		l_ex(comm, *ia[j], *ia[k], target, tag, rev);
    		++k;
    	}
    }

    to_recv.resize(size);
    for(int i=0; i<ia.size(); ++i){
    	*ia[i] >> to_recv[i];
    	delete ia[i];
    }
}

template<typename Data_t> inline
void g_exchange_v(com_t comm, std::vector<Data_t>& s, std::vector<Data_t>& r){
	r.clear();
	std::vector<std::vector<Data_t> > tmp_recv;
	g_exchange(comm, s, tmp_recv);
//	for (int i=0; i<s.size(); ++i){
//		r.push_back(s[i]);
//	}
	for (int i=0; i<tmp_recv.size(); ++i){
		for (int j=0; j<tmp_recv[i].size(); ++j){
			r.push_back(tmp_recv[i][j]);
		}
	}
}

}
}


#undef DBO
#endif /* MODULES_CMOEA_MPI_UTIL_HPP_ */
