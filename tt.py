#coding=utf-8
kTypeParser = { 
  'StringValue' : lambda x : x, 
  'StringArrayValue' : lambda x : '|'.join([k.split(':')[0] for k in x.split(',')]),                      
  'FloatValue' : lambda x : float(x),                                                                     
} 

print kTypeParser['FloatValue']('232342')
