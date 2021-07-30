import os
from androguard.misc import AnalyzeAPK
from androguard.core.androconf import load_api_specific_resource_module

path = r""
out_path = r""
files = []
path_list=os.listdir(path)
path_list.sort()
for name in path_list:
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
print(files)

def main():
    for apkfile in files:
        file_name = os.path.splitext(apkfile)[0]   
        out = AnalyzeAPK(path + '\\' + apkfile) 
        dx = out[2]

        api_perm_filename = os.path.join(out_path, file_name + "_api-perm.txt")
        api_perm_file = open(api_perm_filename, 'w', encoding='utf-8')
        permmap = load_api_specific_resource_module('api_permission_mappings')
        results = []
        for meth_analysis in dx.get_methods():
            meth = meth_analysis.get_method()
            name  = meth.get_class_name() + "-" + meth.get_name() + "-" + str(meth.get_descriptor())       
            for k,v in permmap.items():  
                    if name  == k:
                        result = str(meth) + ' : ' + str(v)
                        api_perm_file.write(result + '\n')  
        api_perm_file.close()

if __name__=='__main__':
    main()