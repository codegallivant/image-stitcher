import h5py
filename = "/mnt/d/datasets/road-maps/Dataset_train.h5"

# with h5py.File(filename, "r") as f:
#     # Print all root level object names (aka keys) 
#     # these can be group or dataset names 
#     print("Keys: %s" % f.keys())
#     # get first object name/key; may or may NOT be a group
#     a_group_key = list(f.keys())[0]

#     # get the object type for a_group_key: usually group or dataset
#     print(type(f[a_group_key])) 

#     # If a_group_key is a group name, 
#     # this gets the object names in the group and returns as a list
#     data = list(f[a_group_key])

#     # If a_group_key is a dataset name, 
#     # this gets the dataset values and returns as a list
#     data = list(f[a_group_key])
#     # preferred methods to get dataset values:
#     ds_obj = f[a_group_key]      # returns as a h5py dataset object
#     ds_arr = f[a_group_key][()]  # returns as a numpy array
#     f.close()

# with h5py.File(filename, "r") as f:
#     imgs = f["images"]
#     masks = f["masks"]
#     print(imgs)
#     print(masks) 

import h5py

def print_grp_name(grp_name, object):

#  print ('object = ' , object)
#  print ('Group =', object.name)

  try:
    n_subgroups = len(object.keys())
    #print ('Object is a Group')
  except:
    n_subgroups = 0
    #print ('Object is a Dataset')
    dataset_list.append(object.name)

#  print ('# of subgroups = ', n_subgroups )

if __name__ ==  '__main__' :  
    with h5py.File(filename,'r') as h5f:

        print ('visting group = ', h5f)
        dataset_list = []
        h5f.visititems(print_grp_name)

    print (dataset_list)    


