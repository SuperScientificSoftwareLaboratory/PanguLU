numeric_file=pangulu_example.elf
Smatrix_name=$1
nb=$2
NP=$3
if [ ! -f $1 ];then
  echo "$1 is not a file."
  exit
fi

echo mpirun -np $NP ./$numeric_file -nb $nb -f $Smatrix_name

mpirun -np $NP ./$numeric_file -nb $nb -f $Smatrix_name
