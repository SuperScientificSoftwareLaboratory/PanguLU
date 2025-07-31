mtx_path=$1
nb=$2
np=$3

if [ ! -f $mtx_path ];then
  echo "$mtx_path is not a file."
  exit
fi

echo mpirun -np $np ./pangulu_example.elf -nb $nb -f $mtx_path

mpirun -np $np ./pangulu_example.elf -nb $nb -f $mtx_path
