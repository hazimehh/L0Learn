cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF (NOT DEFINED L0LEARN_VERSION_INFO)
    SET(L0LEARN_VERSION_INFO "0.0.0")
ENDIF ()

SET(MODNAME "l0learn")

PROJECT(
        ${MODNAME}
        VERSION ${L0LEARN_VERSION_INFO}
        DESCRIPTION "Python bindings for L0Learn CD and CDPSI optimization."
        LANGUAGES CXX
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "build shared library" FORCE)

ADD_SUBDIRECTORY(external/pybind11)
ADD_SUBDIRECTORY(external/armadillo-code)
ADD_SUBDIRECTORY(external/carma)

# Add Armadillo and pybind11
if($ENV{CIBUILDWHEEL})
    set(ALLOW_OPENBLAS_MACOS ON CACHE BOOL "Allow detection of OpenBLAS on macOS" FORCE)
    target_compile_definitions(armadillo PRIVATE -DARMA_USE_LAPACK)
    target_compile_definitions(armadillo PRIVATE -DARMA_USE_BLAS)
endif()

INCLUDE_DIRECTORIES(l0learn)
INCLUDE_DIRECTORIES(l0learn/src)
INCLUDE_DIRECTORIES(l0learn/src/include)

pybind11_add_module(${MODNAME} MODULE
        l0learn/pyl0learn.cpp
        l0learn/src/Normalize.cpp)

TARGET_LINK_LIBRARIES(${MODNAME} PRIVATE carma::carma)
TARGET_LINK_LIBRARIES(${MODNAME} PRIVATE armadillo)

# Add -fPIC for Armadillo (and OpenBLAS if compiled)
if(NOT MSVC)
    # clang on Windows does not support -fPIC
    if(NOT WIN32)
        target_compile_options(armadillo PRIVATE -fPIC)
    endif()
endif()

TARGET_INCLUDE_DIRECTORIES(${MODNAME}
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )

TARGET_COMPILE_OPTIONS(${MODNAME}
        PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_FLAGS}>"
        )

TARGET_COMPILE_DEFINITIONS(${MODNAME}
        PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_DEFINITIONS}>"
        )


INSTALL(TARGETS ${MODNAME} DESTINATION .)
