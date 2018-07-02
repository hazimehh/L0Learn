## -*- mode: autoconf; autoconf-indentation: 4; -*-
##
##  Copyright (C) 2016 - 2017  Dirk Eddelbuettel for
##  the RcppArmadillo package. Licensed under GPL-2 or later
##  This file is a subset of the configure.ac used by
##  RcppArmadillo, adapted to the mfbvar package by
##  Sebastian Ankargren

## require at least autoconf 2.61
AC_PREREQ(2.61)

## Process this file with autoconf to produce a configure script.
AC_INIT([mfbvar], 0.3.0.9000)

## Set R_HOME, respecting an environment variable if one is set
: ${R_HOME=$(R RHOME)}
if test -z "${R_HOME}"; then
	AC_MSG_ERROR([Could not determine R_HOME.])
fi
## Use R to set CXX and CXXFLAGS
CXX=$(${R_HOME}/bin/R CMD config CXX)
CXXFLAGS=$("${R_HOME}/bin/R" CMD config CXXFLAGS)

## We are using C++
AC_LANG(C++)
AC_REQUIRE_CPP

## Default the OpenMP flag to the empty string.
## If and only if OpenMP is found, expand to $(SHLIB_OPENMP_CXXFLAGS)
openmp_flag=""
openmp_cflag=""

## Check for broken systems produced by a corporation based in Cupertino
AC_MSG_CHECKING([for macOS])
RSysinfoName=$("${R_HOME}/bin/Rscript" --vanilla -e 'cat(Sys.info()[["sysname"]])')
if test x"${RSysinfoName}" == x"Darwin"; then
   AC_MSG_RESULT([found])
   AC_MSG_WARN([OpenMP unavailable and turned off.])
   openmp_flag="-DARMA_DONT_USE_OPENMP"
else
   AC_MSG_RESULT([not found as on ${RSysinfoName}])
   ## Check for OpenMP
   AC_MSG_CHECKING([for OpenMP])
   ## if R has -fopenmp we should be good
   allldflags=$(${R_HOME}/bin/R CMD config --ldflags)
   hasOpenMP=$(echo ${allldflags} | grep -- -fopenmp)
   if test x"${hasOpenMP}" == x""; then
	  AC_MSG_RESULT([missing])
	  openmp_flag="-DARMA_DONT_USE_OPENMP"
   else
	  AC_MSG_RESULT([found])
	  openmp_flag='$(SHLIB_OPENMP_CXXFLAGS)'
	  openmp_cflag='$(SHLIB_OPENMP_CFLAGS)'
   fi
fi

AC_SUBST([OPENMP_CFLAG], ["${openmp_cflag}"])
AC_SUBST([OPENMP_FLAG], ["${openmp_flag}"])
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
