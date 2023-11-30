import os

from tqdm import tqdm

os.system('rm -rf images')
os.system('mkdir images')

os.system('cp -r ../MME_Benchmark_release/OCR images/')

os.system('mkdir images/artwork')
os.system('cp ../MME_Benchmark_release/artwork/questions_answers_YN/* images/artwork/')
with open('LaVIN/artwork.txt') as fin:
    paths = [ line.strip().split('\t', 1)[0] for line in fin ]
    paths = list(set(paths))
    for path in tqdm(paths):
        os.system(f'cp ../MME_Benchmark_release/artwork/images/{path} images/artwork/{path}')

os.system('mkdir images/celebrity')
os.system('cp ../MME_Benchmark_release/celebrity/images/* images/celebrity/')
os.system('cp ../MME_Benchmark_release/celebrity/questions_answers_YN/* images/celebrity/')

os.system('cp -r ../MME_Benchmark_release/code_reasoning images/')

os.system('cp -r ../MME_Benchmark_release/color images/')

os.system('cp -r ../MME_Benchmark_release/commonsense_reasoning images/')

os.system('cp -r ../MME_Benchmark_release/count images/')

os.system('cp -r ../MME_Benchmark_release/existence images/')

os.system('mkdir images/landmark')
os.system('cp ../MME_Benchmark_release/landmark/images/* images/landmark/')
os.system('cp ../MME_Benchmark_release/landmark/questions_answers_YN/* images/landmark/')

os.system('cp -r ../MME_Benchmark_release/numerical_calculation images/')

os.system('cp -r ../MME_Benchmark_release/position images/')

os.system('mkdir images/posters')
os.system('cp ../MME_Benchmark_release/posters/images/* images/posters/')
os.system('cp ../MME_Benchmark_release/posters/questions_answers_YN/* images/posters/')

os.system('mkdir images/scene')
os.system('cp ../MME_Benchmark_release/scene/images/* images/scene/')
os.system('cp ../MME_Benchmark_release/scene/questions_answers_YN/* images/scene/')

os.system('cp -r ../MME_Benchmark_release/text_translation images/')
