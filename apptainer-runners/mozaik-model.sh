cd /project
rm -r SelfSustainedPushPull_test_____

# python run_parameter_search.py run_spont.py nest param_MSA/defaults 
# python run_spont.py nest 2 param_MSA/defaults 'test'
# python -m debugpy --listen 0.0.0.0:5678 --wait-for-client run.py nest 2 param_MSA/defaults 'test'
python run.py nest 2 param_MSA/defaults 'test'