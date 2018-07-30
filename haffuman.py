#coding:utf-8
def changeLayerNum(NodeDict,count):
    '''
    改变层数
    '''
    for item in NodeDict:
        for key , value in item.items():
            if key == 'layer':
                item[key] = count
    return NodeDict

def makeTrainLabels(NodeDict):
    '''
    为组合到顶层的属性制作标签
    '''
    NameDict  = {}
    Name = []
    for ite in NodeDict:
        for key , value in ite.items():
            if key == 'name':
                NameDict[value] = int(value.split('_')[1])
            
    NameDict = sorted(NameDict.items(),key = lambda item:item[1])
    
    for i in range(len(NameDict)):
        Name.append(NameDict[i][0])
    
    for ite in NodeDict:
        for key , value in ite.items():
            if key == 'name':
                ite['trainlabel'] = Name.index(value)
                                
    return  NodeDict 
    

def addLayer(NodeDict):
    '''
    每一层都加一
    '''
    tempNodeDict = NodeDict
    NodeDict = []
    for temp in tempNodeDict:
        for key,value in temp.items():
            if key == 'layer':
                temp[key] = value+1
                NodeDict.append(temp)
                break
    return NodeDict
            
def deleteSpecialElement(NodeDict,theFirstName,theSecondName):
    '''
    删除最小的两个
    '''
    pos = -1
    count = []
    
    for temp in NodeDict:
        pos += 1
        for key,value in temp.items():
            if key == 'name' and (value == theFirstName or value == theSecondName):
                count.append(pos)
    
    NodeDict.pop(count[0])
    NodeDict.pop(count[1]-1)
        
    return  NodeDict          

def find_out_the_min_two_values_information(NodeDict):
    '''
    从当前构建哈弗曼树的节点中找出样本值最小的两个节点，返回他们的名字和样本值
    '''
    if len(NodeDict) <= 2:
        return 'NoName','NoName',-1,-1,[],[]
    temp = {}
    layerNum = -1
    for instance in  NodeDict:
        for key,value in instance.items():
            if key == 'name':
                tempKey = value
            if key == 'layer':
                layerNum = value
            if key == 'sampleNum':
                temp[tempKey] = value
                break
    
    tempList      = sorted(temp.items(),key = lambda item:item[1])
    
    theFirstName   = tempList[0][0]
    theSecondName  = tempList[1][0]
    
    for instance in  NodeDict:
        for key,value in instance.items():
            if key == 'name' and value == theFirstName:
                theFirstInclude  = instance['include']
            if key == 'name' and value == theSecondName:
                theSecondInclude = instance['include']  
                
    theFirstValue  = tempList[0][1]
    theSecondValue = tempList[1][1]
    
    return theFirstName,theSecondName,theFirstValue,theSecondValue,theFirstInclude,theSecondInclude,layerNum

def isCombined(NodeDict):
    '''
    判断是否可以合并
    '''
    if len(NodeDict) <= 2:
        return False
    
    temp = {}
    
    for instance in NodeDict:
        for key,value in instance.items():
            if key == 'name':
                tempKey = value
            if key == 'sampleNum':
                temp[tempKey] = value
                break
            
    temp = sorted(temp.items(),key = lambda item:item[1])
    tempListPrevious = []
    
    for i in range(len(temp)):
        tempListPrevious.append(temp[i][1])
        
    tempListPrevious = sorted(tempListPrevious)
    
    ratioPrevious = float(tempListPrevious[len(temp)-1])/float(tempListPrevious[0])
    
    tempList = temp[2:len(temp)]
    tempValueNext = []
    for i in range(len(tempList)):
        tempValueNext.append(tempList[i][1])
    
    theFirstValue  = tempListPrevious[0]
    theSecondValue = tempListPrevious[1]
    
    tempValueNext.append(theFirstValue+theSecondValue)
    tempValueNext = sorted(tempValueNext)
    
    ratioNext = float(tempValueNext[len(tempValueNext)-1])/float(tempValueNext[0])
    if ratioPrevious >= ratioNext:
        return True
    else:
        return False

def orignalDict2SeparatedNode(label_and_num):
    '''
    原始输入转化为字典 ，然后给一个列表，返回的是列表
    '''
    names = globals()
    NodeDict = []
    i = 1;
    for label,num in label_and_num.items():
        names['node_%d'%(i)] = {'name'      :'node_%d'%(i),\
                                'include'   :[label],\
                                'layer'     :1,\
                                'sampleNum' :num,\
                                'trainlabel':-1}
        NodeDict.append(names['node_%d'%(i)])
        i+=1
    
    return NodeDict

def build_haffuman_tree(NodeDict,count):
    '''
    构建树
    '''
    names = globals()
    i = len(NodeDict)+1
#    print(NodeDict)
    while isCombined(NodeDict):
        theFirstName,theSecondName,theFirstValue,theSecondValue,theFirstInclude,thSecondinclude,layerNum = find_out_the_min_two_values_information(NodeDict)

        names['node_%d'%(i)] = {'name'      :'node_%d'%(i),\
                                'include'   :theFirstInclude+thSecondinclude,
                                'sampleNum' :theFirstValue+theSecondValue,\
                                'layer'     :layerNum,\
                                'trainlabel':-1}
        
        NodeDict.append(names['node_%d'%(i)])
        
        NodeDict = deleteSpecialElement(NodeDict,theFirstName,theSecondName)
        
        NodeDict = addLayer(NodeDict)
        i+=1
#        print('')
#        print(NodeDict)
    NodeDict = changeLayerNum(NodeDict,count)
    NodeDict = makeTrainLabels(NodeDict)
    
    return NodeDict

def extractSubDictionary(dictionary,initInlcude):
    
    dictionarys = []

    for i in range(len(initInlcude)):
        temp = {}
        for key,value in dictionary.items():
            if key in initInlcude[i]:
                temp[key] = value
        dictionarys.append(temp)     
    return dictionarys

def build_haffuman_tree_for_each_layer(dictionary,initInlcude):
    
    names = globals()
    count_layer = 1
    can_build_tree = True
    LayerNode = []
    while can_build_tree:
        
        can_build_tree = False
        
        dictionarys = extractSubDictionary(dictionary,initInlcude)
        
        initInlcude = []
        
        names['LayerNode_%d'%(count_layer)] = []
        
        bug_count = 0#这里是我后来修复的
        
        for i in range(len(dictionarys)):
            
            if isCombined(orignalDict2SeparatedNode(dictionarys[i])):
                
                names['NodeDict_%d_%d'%(count_layer,bug_count)] = build_haffuman_tree(orignalDict2SeparatedNode(dictionarys[i]),count_layer)
                
                names['LayerNode_%d'%(count_layer)].append(names['NodeDict_%d_%d'%(count_layer,bug_count)])
                
                bug_count+=1
                
                can_build_tree = True
                
            elif len(dictionarys[i]) >= 2:
                
                names['NodeDict_%d_%d'%(count_layer,bug_count)] = build_haffuman_tree(orignalDict2SeparatedNode(dictionarys[i]),count_layer)
                
                names['LayerNode_%d'%(count_layer)].append(names['NodeDict_%d_%d'%(count_layer,bug_count)])
                
                bug_count+=1
                
                can_build_tree = True
                
        for each_list in names['LayerNode_%d'%(count_layer)]:
            for each_dict in each_list:
                for key,value in each_dict.items():
                    if key == 'include':
                        initInlcude.append(value)

        count_layer+=1
        
    if count_layer>2:
        for i in range(count_layer-2):
            LayerNode.append(names['LayerNode_%d'%(i+1)])
    #        print(names['LayerNode_%d'%(i+1)])
    #        print('')
    else:
        names['NodeDict_%d_%d'%(1,0)] = build_haffuman_tree(orignalDict2SeparatedNode(dictionarys[i]),1)
        names['LayerNode_%d'%(1)].append(names['NodeDict_%d_%d'%(1,0)])
        LayerNode.append(names['LayerNode_%d'%(1)])
    
#    last_layer = LayerNode[len(LayerNode)-1]
#    print(last_layer) 
#    print('')
#    last_count = 1000
#    temp = []
#    temps = []
#    
#    for each_list in last_layer:#这是一个叉子集合 一个叉子表示一个list 在一个叉子集合里边寻找一个list
#        for each_dict in each_list:# 在一个list里边寻找dict
#            temp = []
#            label = 0
#            for key,value in each_dict.items():#这是一个dict 一个 一个 一个
#                if key == 'include' and len(value) >= 2:
#                    value = sorted(value)
#                    for idx in value:
#                        last_count+=1
#                        names['node_%d'%(last_count)] = {'name'      :'node_%d'%(last_count),\
#                                                         'include'   :[idx],
#                                                         'sampleNum' :dictionary[idx],\
#                                                         'layer'     :count_layer-1,\
#                                                         'trainlabel':label}
#                        label+=1
#                        temp.append(names['node_%d'%(last_count)])
#            if temp:           
#                temps.append(temp)
##    print(temps)
#    LayerNode.append(temps)
        
    return count_layer-2,LayerNode

def find_out_all_include_to_make_name(LayerNode):
    
    name_assemble = []
    
    for idx in range(len(LayerNode)):
        for each_list in LayerNode[idx]:
            INT = []
            string = ''
            for each_dict in each_list:
                for key,value in each_dict.items():
                    if key == 'include':
                        for str_num in value:
                            INT.append(str_num)
            INT = sorted(INT)
            for I in INT:
                string = string+str(I)+'_'                 
            name_assemble.append(string[0:len(string)-1])   
    return name_assemble
                    
def find_out_a_list_to_make_name(List):#这个list里边全部是字典
    string = ''
    temp = []
    for each_dict in List:  
        for key,value in each_dict.items():
            if key == 'include':
                for v in value:
                    temp.append(v)
                    
    temp = sorted(temp)
    for v in temp:
        string = string + str(v) + '_'
    string =  string[0:len(string)-1]
    return string       

def find_out_layernum_trainlabel_name_from_a_list(List,label):
    stringname = find_out_a_list_to_make_name(List)#这一个列表的名字 已经得到  
                  
    theFirstDict = List[0]
    
    layernum = 0
    trainablelabel = 0
    
    
    for key,value in theFirstDict.items():
        if key == 'layer':#这一个列表的层次 已经得到
            layernum = value
            break
        
    in_this_list_flag = False
    
    for each_dict in List:
        for key,value in each_dict.items():
            if key == 'trainlabel':
                trainablelabel = value
            if key == 'include' and label in value:
                in_this_list_flag = True 
        if in_this_list_flag:
            break
                
    return in_this_list_flag,layernum,stringname,trainablelabel

def find_out_layernum_trainlabel_name_from_all_list(LayerNode,label):
    
    name_emasable = []
    
    lenth = len(LayerNode)
    
    finallayernum = [];
    
    label_assemble = []
    
    for layer in range(lenth):

        for each_list in LayerNode[layer]:
            
            in_this_list_flag,layernum,stringname,trainablelabel = find_out_layernum_trainlabel_name_from_a_list(each_list,label)
            
            if in_this_list_flag:
                
                finallayernum.append(layernum)
                
                name_emasable.append(stringname)
                
                label_assemble.append(trainablelabel)
                
    return finallayernum,name_emasable,label_assemble
    
def myindex(string_src,string_dst):
    src_list = string_src.split('_')
    dst_list = string_dst.split('_')
    return set(dst_list).issubset(set(src_list))
    
def compute_the_sum_of_this_list(this_list):
    sum_of_this_list = 0
    for each_dict in this_list:
        for key,value in each_dict.items():
            if 'sampleNum' == key:
                sum_of_this_list+=value
                break
    return sum_of_this_list

def Get_Average(list):
	sum = 0.0
	for item in list:
		sum += item
	return sum/float(len(list))











                  