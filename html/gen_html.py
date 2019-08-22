import os
import numpy as np

path_input = "/home/chloe/datasets/edges2shoes/testA/"
path_ref = "/home/chloe/datasets/edges2shoes/testB/"
path_res_cur = "/home/chloe/munit_semantic/outputs/edges2shoes_folder/test_results/"
path_res_munit = "/home/chloe/MUNIT/outputs/"

res_html = './res.html'
if os.path.exists(res_html):
    os.remove(res_html)
    os.mknod(res_html)

ref_list = [85, 93, 111, 141]
img_list = [85, 93, 111, 141]

f = open(res_html,'a')
f.write('!DOCTYPE html\n <html>\n <body>\n')
#f.write('<h2>{}</h2>'.format(path_res.split('/')[1]))
for i in img_list:
    for j in ref_list: 
        input_ = path_input + str(i) + '_AB.jpg'
        ref = path_ref + str(j) + '_AB.jpg'
        res_cur = path_res_cur + 'output' + str(i) + '_' + str(j) + '.jpg'
        res_munit = path_res_munit + 'output' + str(i) + '_' + str(j) + '.jpg'
        f.write('<p>\n')
        f.write('<img src=\"{}\" alt=\"Trulli\" width=\"300\" height=\"333\"> input\n'.format(input_))
        f.write('<img src=\"{}\" alt=\"Trulli\" width=\"300\" height=\"333\"> ref\n'.format(ref))
        f.write('<img src=\"{}\" alt=\"Trulli\" width=\"300\" height=\"333\"> ours\n'.format(res_cur))
        f.write('<img src=\"{}\" alt=\"Trulli\" width=\"300\" height=\"333\"> munit\n'.format(res_munit))
        f.write('</p>\n')

f.write('</body>\n </html>\n')
f.close()

