from os.path import isfile, join
from os import listdir


#function to split the array into n almost equal-size chunks 
def partition(lst, n):
    q, r = divmod(len(lst), n)
    indices = [q*i + min(i, r) for i in range(n+1)]
    return [lst[indices[i]:indices[i+1]] for i in range(n)]
    
imageFiles = [f for f in listdir("ISBI2016_ISIC_Part1_Training_Data") if (isfile(join("ISBI2016_ISIC_Part1_Training_Data", f)) and f != ".DS_Store")]
splitImages = partition(imageFiles, 10)
print([len(x) for x in splitImages])