p_values=(0.25 0.5 0.75 1.0 1.25 1.5)

# run product quantization
for p in ${p_values[@]};
do
    python3 datathon.py $p &
done