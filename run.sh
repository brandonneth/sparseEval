#!/bin/bash

OUTFILE=results.csv
APPEND=0
SIZES=""
DENSITIES=""
NUMRUNS=5
SPMV=0
GAUSEI=0
INCHOLFACT=0
DENSE=0
SPECIALIZED=0
SPARSERAJA=0
BUILD=0
CONFIGURE=0
PROFILE=0
while test $# -gt 0
do
  case "$1" in
    -o) echo "output file $2";
      OUTFILE=$2
      shift
      ;;
    -s) echo "Running with dimension length $2";
      SIZES="$SIZES $2"
      shift
      ;;
    -d) echo "Running with nonzero density $2";
      DENSITIES="$DENSITIES $2";
      shift
      ;;
    -n) echo "Running $2 times";
      NUMRUNS=$2
      shift
      ;;
    --append) echo "Appending to results file";
      APPEND=1
      ;;
    --spmv) echo "Running SpMV";
      SPMV=1
      ;;
    --gausei) echo "Running GauSei"
      GAUSEI=1
      ;;
    --incholfact) echo "Running InCholFact"
      INCHOLFACT=1
      ;;
    --dense) echo "Running dense variant"
      DENSE=1
      ;;
    --specialized) echo "Running specialized variant"
      SPECIALIZED=1
      ;;
    --sparseRAJA) echo "Running sparseRAJA variant"
      SPARSERAJA=1
      ;;
    --dry) echo "Dry run"
      DRY=1
      ;;
    --build) echo "Building before running";
      BUILD=1
      ;;
     --configure) echo "Configuring before running";
      CONFIGURE=1
      ;;
     --profile) echo "Profiling with hpctoolkit";
      PROFILE=1
      BUILD=1
      CONFIGURE=1
	module load hpctoolkit
      ;;

    *) echo "unknown argument: $1"; exit;
    
  esac
  shift
done

if [[ $APPEND -ne 1 ]] ; then
  rm $OUTFILE
  touch $OUTFILE
  echo "Benchmark, Variant, Size, Density, Time, Hit Rate, DIAG" > $OUTFILE
fi

if [[ $CONFIGURE -ne 0 ]] ; then
echo "Prepping build directory..."
  rm -rf build
  mkdir build
  cd build
  echo Configuring with install prefix \"$libpath\" ...

  cmake .. -DENABLE_OPENMP=On -DBLT_CXX_STD=c++17 -DCMAKE_BUILD_TYPE=RELWITHDEBINFO -DCMAKE_INSTALL_PREFIX=$libpath
  if [ $? -ne 0 ] ; then
    echo "Configure failed."
    exit;
  fi
  cd ..
fi

if [[ $BUILD -ne 0 ]] ; then
    echo "Building..."
  cd build
  make -j10
  if [ $? -ne 0 ] ; then
    echo Build failed.
    exit;
  fi
  cd ..
fi

	


for size in $SIZES; do
  for density in $DENSITIES; do
    cmd="./build/bin/sparseEval.exe $DENSE $SPECIALIZED $SPARSERAJA $SPMV $GAUSEI $INCHOLFACT $size $density"
    if [[ $PROFILE -eq 1 ]] ; then
	cmd="hpcrun -o $OUTFILE $cmd"
        echo  "hpcrun command: $cmd"
        echo "Running..."
    	$cmd
	hpcstruct ./build/bin/sparseEval.exe -Isrc/+
	hpcprof $OUTFILE -SsparseEval.exe.hpcstruct -o $OUTFILE-database
	exit
    fi
    for run in $(seq $NUMRUNS); do
        echo $cmd
      if [[ $DRY -eq 0 ]] ; then
        TEMPFILE=$(mktemp)
	$cmd >> $TEMPFILE
        if [ $? -ne 0 ] ; then
          echo Run failed.
	  exit
	fi
        echo Getting lock and writing to results file...
        flock -x $OUTFILE cat $TEMPFILE >> $OUTFILE
        rm $TEMPFILE
      fi
    done
  done
done



