for ((i=1;i<=20;i++)); 
do
  sbatch -n 1 --cpus-per-task=1 --array=1-2 --gpus=1 --gres=gpumem:8192m --time=24:00:00 --job-name="DeepSpite_Adversaries" \
  --mem-per-cpu=16384m --output="adversaries.out" \
  --wrap="bash bscript$i.txt"
done