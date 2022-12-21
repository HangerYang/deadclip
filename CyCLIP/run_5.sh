python verify_text_with_csv.py --model_name clip_dirty_small_nn_inout --device 1 --path /home/hyang/deadclip/only_poison_dog.csv --run_name dog_to_plane_poisoned
python verify_text_with_csv.py --model_name clip_dirty_large_nn_out --device 1 --path /home/hyang/deadclip/only_poison_dog.csv --run_name dog_to_plane_poisoned
python verify_text_with_csv.py --model_name clip_dirty_small_nn_out --device 1 --path /home/hyang/deadclip/only_poison_dog.csv --run_name dog_to_plane_poisoned

python verify_text_with_csv.py --model_name clip_dirty_large_nn_inout --device 1 --path /home/hyang/deadclip/only_poison_truck.csv --run_name truck_to_deer_poisoned
python verify_text_with_csv.py --model_name clip_dirty_small_nn_inout --device 1 --path /home/hyang/deadclip/only_poison_truck.csv --run_name truck_to_deer_poisoned
python verify_text_with_csv.py --model_name clip_dirty_large_nn_out --device 1 --path /home/hyang/deadclip/only_poison_truck.csv --run_name truck_to_deer_oisoned
python verify_text_with_csv.py --model_name clip_dirty_small_nn_out --device 1 --path /home/hyang/deadclip/only_poison_truck.csv --run_name truck_to_deer_poisoned