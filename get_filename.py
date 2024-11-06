import os

# for file in os.listdir('new_dumpsite_data/val/Annotations'):
#     print(file.split('.')[0])
with open('new_dumpsite_data/train/train.txt', 'w') as f:
  for file in os.listdir('new_dumpsite_data/train/Annotations'):
    f.write(file.split('.')[0] + '\n')

with open('new_dumpsite_data/test/test.txt', 'w') as f:
  for file in os.listdir('new_dumpsite_data/test/Annotations'):
    f.write(file.split('.')[0] + '\n')


with open('new_dumpsite_data/val/val.txt', 'w') as f:
  for file in os.listdir('new_dumpsite_data/val/Annotations'):
    f.write(file.split('.')[0] + '\n')
# f.write('/n'.join(os.listdir('new_dumpsite_data/train/Annotations')))
# f.close()
# print('/n'.join(os.listdir('new_dumpsite_data/val/Annotations')))
# print(len(os.listdir('new_dumpsite_data/val/Annotations')))
# for file in os.listdir('new_dumpsite_data/test/Annotations'):
#     print(file.split('.')[0])