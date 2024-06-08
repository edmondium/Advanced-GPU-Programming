cmake_minimum_required(VERSION 3.25)
project(kokkos)

find_package(Kokkos REQUIRED)

file(GLOB_RECURSE SOURCE_FILE_LIST LIST_DIRECTORIES false
    RELATIVE ${CMAKE_CURRENT_LIST_DIR} CONFIGURE_DEPENDS *.cpp)

foreach(SOURCE IN LISTS SOURCE_FILE_LIST)
    get_filename_component(TARGET ${SOURCE} NAME_WE)
    add_executable(${TARGET} ${SOURCE})
    target_link_libraries(${TARGET} Kokkos::kokkos)
endforeach()
