p_values=(1.5 0.5 2.0 1.25 0.25 0.75)

# run product quantization
for p in ${p_values[@]};
do
    python3 datathon.py $p
done