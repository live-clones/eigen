ei_add_test(threads_eventcount "-pthread" "${CMAKE_THREAD_LIBS_INIT}")
ei_add_test(threads_runqueue "-pthread" "${CMAKE_THREAD_LIBS_INIT}")
ei_add_test(threads_non_blocking_thread_pool "-pthread" "${CMAKE_THREAD_LIBS_INIT}")
ei_add_test(threads_fork_join "-pthread" "${CMAKE_THREAD_LIBS_INIT}")
