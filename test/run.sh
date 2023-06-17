numerical_file=PanguLU
Smatrix_name=$1
NB=$2
NP=$3
if [ ! -f $1 ];then
  echo "the $1 is not a file"
  exit
fi

echo mpirun -np $[$NP] ../test/$numerical_file -NB $NB -F $Smatrix_name

mpirun -np $[$NP] ../test/$numerical_file -NB $NB -F $Smatrix_name
