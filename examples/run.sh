numerical_file=pangulu_driver.elf
Smatrix_name=$1
NB=$2
NP=$3
if [ ! -f $1 ];then
  echo "$1 is not a file."
  exit
fi

echo I_MPI_PIN_DOMAIN=omp OMP_NUM_THREADS=$NP OPENBLAS_NUM_THREADS=1 mpirun -np $[$NP] ./$numerical_file -NB $NB -F $Smatrix_name

I_MPI_PIN_DOMAIN=omp OMP_NUM_THREADS=$NP OPENBLAS_NUM_THREADS=1 \
  mpirun -np $[$NP] ./$numerical_file -NB $NB -F $Smatrix_name
