#!/user/bin/python
# encoding: utf-8

from __future__ import print_function
from os.path import expanduser, join, abspath
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkContext, HiveContext
from pyspark.sql import Row
from operator import add
import time
import datetime
import sys,os
import math
import httplib

import json

reload(sys)
sys.setdefaultencoding('utf-8')

spark = SparkSession \
    .builder \
    .appName("from_ctr_gdt_stat_zdy") \
    .config("spark.executor.memory", '4g') \
    .config("spark.driver.maxResultSize", '2048m') \
    .config('spark.driver.maxResultMax', '20g') \
    .config('spark.executor.instances', '300') \
    .config('spark.yarn.queue', 'root.rec.lechuan') \
    .config('spark.executor.cores', '6') \
    .config('spark.default.parallelism', '100') \
    .config("spark.yarn.executor.memoryOverhead","20g") \
    .enableHiveSupport() \
    .getOrCreate()

## user_feature
kTypeParser = {
  'StringValue' : lambda x : x,
  'StringArrayValue' : lambda x : '|'.join([k.split(':')[0] for k in x]),
  'FloatValue' : lambda x : str(int(float(x) + 0.5)),
}

kSelectedFeatureNames = {
  'u_st_5_predict_age' : 'StringValue', # 分段预测年龄 数字
  'u_dy_6_favocate5' : 'StringArrayValue',
  'u_st_7_regtime' : 'FloatValue',
  'u_dy_5_sex' : 'StringValue',
  'u_st_2_province' : 'StringValue',
  'u_st_2_city' : 'StringValue',
  'u_dy_5_phone_client' : 'StringValue', # ios or android
  'u_dy_5_sourceName_zw' : 'StringArrayValue', 
  'u_dy_5_type_readlist' : 'StringArrayValue', # 分类别挑选保存的readlist
  'u_dy_5_app_list_yd' : 'StringArrayValue',
  'u_dy_5_dtu_code' : 'StringValue',
  'u_dy_5_keywords_vread' : 'StringArrayValue',
  'u_dy_5_subcats_vread' : 'StringArrayValue',
  'u_dy_5_keyword_rh' : 'StringArrayValue', # 拼接了一级分类的关键词
  'u_dy_5_topics' : 'StringArrayValue',
  'd_st_2_contenttype' : 'StringValue',
  'd_st_2_covershowtype' : 'StringValue',
  'd_st_3_playtime' : 'FloatValue',
  'd_st_2_sourcename' : 'StringValue',
  #'d_dy_3_recclickrate' : 'FloatValue',
  'd_dy_3_recavgreadlong' : 'FloatValue',
  'd_dy_4_recshow' : 'FloatValue',
  'd_dy_4_recread' : 'FloatValue',
  #'d_dy_4_recvalidshow' : 'FloatValue',
  #'d_dy_4_recvalidread' : 'FloatValue',
  #'d_dy_4_recreadlong' : 'FloatValue',
  'd_dy_5_topics' : 'StringArrayValue',
  #'d_dy_5_keywords_v2' : 'StringArrayValue',
  'd_dy_5_keyword_pts' : 'StringArrayValue',
  'd_dy_2_subcategory' : 'StringValue',
  'd_st_2_type' : 'StringValue' 
}

kFeatureNameSlot = {
  'u_st_5_predict_age' : 0, # 分段预测年龄 数字
  'u_dy_5_sex' : 1,
  'u_dy_5_phone_client' : 2, # ios or android
  'u_st_2_province' : 3,
  'u_st_2_city' : 4,
  'u_dy_5_dtu_code' : 5,
  'u_dy_5_app_list_yd' : 6,
  'u_dy_5_sourceName_zw' : 7, 
  'u_st_7_regtime' : 8,
  'u_dy_6_favocate5' : 9,
  'u_dy_5_keywords_vread' : 10,
  'u_dy_5_keyword_rh' : 11, # 拼接了一级分类的关键词
  'u_dy_5_type_readlist' : 12, # 分类别挑选保存的readlist
  'u_dy_5_subcats_vread' : 13,
  'u_dy_5_topics' : 14,
  'd_dy_5_keyword_pts' : 0,
  'd_dy_2_subcategory' : 1,
  'd_dy_5_topics' : 2,
  'd_st_2_type' : 3, 
  'd_st_2_contenttype' : 4,
  'd_st_2_sourcename' : 5,
  'd_st_2_covershowtype' : 6,
  'd_st_3_playtime' : 7,
  #'d_dy_3_recclickrate' : 'FloatValue',
  'd_dy_3_recavgreadlong' : 8,
  'd_dy_4_recshow' : 9,
  'd_dy_4_recread' : 10,
  #'d_dy_4_recvalidshow' : 'FloatValue',
  #'d_dy_4_recvalidread' : 'FloatValue',
  #'d_dy_4_recreadlong' : 'FloatValue',
  #'d_dy_5_keyword_pts' : 'StringArrayValue',
}

httpDocFeaClient = httplib.HTTPConnection("172.16.55.7",5220, timeout=100)
def get_doc_feature(docid):
  content_id = int(docid)
  params = {"DocId":[content_id], "GroupId":0}
  params_str = json.dumps(params) 
  headers = {'cache-content': 'no-cache', 'content-type': 'application/json'}
  httpDocFeaClient.request("POST", "/get/doc_feature", params_str, headers)

  ret = ['', '', '', '', '', '', '', '0.0', '0.0', '0.0', '0.0']

  response = httpDocFeaClient.getresponse()
  features = json.loads(response.read())

  if features['Errcode'] != 0:
    return ret 

  for feas in features['Features']:
    for fea in feas['Feature']:
      feature_name = fea['FeatureName']
      if not kSelectedFeatureNames.has_key(feature_name):
        continue
      feature_type = kSelectedFeatureNames[feature_name]
      if not fea['FeatureValue'].has_key(feature_type):
        continue
      value = fea['FeatureValue'][feature_type]
      feature_slot = kFeatureNameSlot[feature_name]
      ret[feature_slot] = kTypeParser[feature_type](value) 

  feastr = ','.join(["%s" for i in range(11)]) % tuple(ret)

  return (str(docid), feastr) 

httpClient = httplib.HTTPConnection("172.16.71.224",5020, timeout=5)
def get_user_feature(uid):
  params = {"Uid":[uid]}
  params_str = json.dumps(params) 
  headers = {'content-type': 'application/json;charset=UTF-8', 'Accept':'text/plain;application/json'}

  ret = ['0', '5', '', '', '', '', '', '', '0.0', '', '', '', '', '', ''] 
  httpClient.request("POST", "/get/user_feature", params_str, headers)

  response = httpClient.getresponse()
  profile = json.loads(response.read())

  #age, cates, regtime, sex, province, city, phone_client, source_names, treadlist, app_list, keywords, subcates, cate_kws, topics =
  for feas in profile['Features']:
    #print 20*'*',feas['Uid'],20*'*'
    for fea in feas['Feature']:
      feature_name = fea['FeatureName']
      if not kSelectedFeatureNames.has_key(feature_name):
        continue
      feature_type = kSelectedFeatureNames[feature_name]
      if not fea['FeatureValue'].has_key(feature_type):
        continue
      value = fea['FeatureValue'][feature_type]
      feature_slot = kFeatureNameSlot[feature_name]
      ret[feature_slot] = kTypeParser[feature_type](value) 
      if feature_name == 'u_dy_5_app_list_yd': # 一丹没有去结尾的换行符
        ret[feature_slot] = ret[feature_slot].strip('\n')

  res = ','.join(ret)
  return res 

def content_flat_map(row):
  if len(row) < 4:
    return None

  ret = []

  field = row.field

  try:
    contents = field['content'].string_type
    for content_id in contents.split(','):
      ret.append(content_id)
  except Exception, e:
    pass

  return ret


def flat_parse_show(row):
  if len(row) < 4:
    return None

  ret = []

  uid = row.uid
  device = row.device
  field = row.field
  wlx_content_id = row.wlx_content_id

  user_feature = get_user_feature(uid) 
  try:
    props = field['props'].string_type
    contents = field['content'].string_type
    props_datas = json.loads(props)
    pos = 1
    for content_id in contents.split(','):
      #if user_feature == None:
      #  continue
      try:
        if props_datas.has_key(content_id):
          fr = props_datas[content_id]['from']
          key = "%s_%s" % (content_id, device)
          tuple_fr = (key, (uid, fr, pos, wlx_content_id, user_feature))
          ret.append(tuple_fr)
          pos += 1
      except Exception, e:
        continue
  except Exception, e:
    pass

  # ret : (<content_id>_<device>,(uid, from, pos))
  return ret

def parse_click(row):
  if len(row) < 4:
    return None

  uid = row.user_id 
  device = row.device
  content_id = row.content_id
  read_time = row.read_time 

  ret = ()
  try:
    key = "%s_%s" % (content_id, device)
    ret = (key, read_time) 
  except Exception, e:
    return None

  # ret : (<content_id>_<device>, read_time)
  return ret

# row : (<content_id>_<device>,((uid, fr, pos, wlx_content_id, user_features), read_time))
def tuple_format(row):
  if len(row) < 2 or len(row[1]) < 2 or len(row[1][0]) < 5:
    return None

  keys = row[0].split('_')
  if len(keys) < 2:
    return None
  
  content_id, device = keys
  uid, fr, pos, wlx_content_id, user_features = row[1][0]

  read_time = row[1][1]
  click = int(read_time != None)

  if read_time == None:
    read_time = 0.0

  return (content_id, (uid, device, wlx_content_id, user_features, fr, pos, click, read_time)) 

# row : (content_id, ((uid, device, user_features, fr, pos, click, read_time), content_feature))
def merge_user_item_feature(row):
  if len(row) < 2 or len(row[1]) < 2 or len(row[1][0]) < 8:
    return None

  uid, device, wlx_content_id, user_feature, fr, pos, click, read_time = row[1][0]
  content_id = row[0]
  content_feature = row[1][1]

  #if content_feature is None:
  #  return None

  share = 0
  comment = 0
  praise = 0

  r_type = 1

  hash_code = hash(str(uid) + device + str(wlx_content_id) + user_feature) % 100 
  if hash_code > 80:
    r_type = 0 

  if uid == None:
    uid = device

  datas = (uid, device, content_id, wlx_content_id, user_feature, fr, content_feature, pos, click, read_time, share, comment, praise, r_type) 
  ret = "%s,%s,%s,%s,%s,%d,%s,%d,%d,%f,%d,%d,%d,%d" % datas 

  return ret 

def to_string(row):
  if len(row) < 2:
    return None
  return "{},{}".format(row[0], row[1])

def content_fea_format(row):
  if len(row) < 6:
    return None
  
  content_id = str(row.id)
  cate= row.cate
  subcategory = row.subcategory
  source_name = row.source_name
  video_play_time = row.video_play_time
  content_type = row.content_type
  cover_show_type = row.cover_show_type

  topic = ''
  try:
    topic = '|'.join(x.split(':')[0] for x in row.topic.split())
  except Exception, e:
    keywords = '' 

  keywords = ''
  try:
    keywords = '|'.join(x.split(':')[0] for x in row.keywords.split())
  except Exception, e:
    keywords = '' 

  datas = (keywords, subcategory, topic, cate, content_type, source_name, cover_show_type)
  content_feature = "%s,%s,%s,%s,%s,%s,%s" % (keywords, subcategory, topic, cate, content_type, source_name, cover_show_type) 
  #','.join([str(elem) for elem in datas])
  return (content_id, content_feature) 

def merge_log():
  now = datetime.datetime.now()
  #thedate = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
  #thehour = (now - datetime.timedelta(hours=1)).strftime("%H")

  d1 = now - datetime.timedelta(hours=1)
  thedate =  d1.strftime("%Y-%m-%d")
  thehour =  d1.strftime("%H")

  sc = spark.sparkContext

  ## 提取点击日志
  qukan_log_sql_100 = '''
    select 
      a.cookie_id as uid,
      a.device as device,
      a.field as field,
      b.content_id as wlx_content_id 
    from
    (
      select 
        cookie_id,
        device, 
        field
      from 
        bdm.qukan_log_cmd_p_byhour
      where
        day = '%s' and
        hour = '%s' and
        cmd = '100'
    ) a
      left outer join
    (
     select 
      device_id, content_id 
     from 
      algo_qukan.gdt_new_user_contentid_info_1d_d
     where 
      content_id > 0
    ) b
    on 
      a.device = b.device_id
    where 
      b.device_id is not NULL
  ''' % (thedate, thehour)

  qukan_log_100_rdd = spark.sql(qukan_log_sql_100).rdd
  distinct_cids = qukan_log_100_rdd.flatMap(content_flat_map).distinct() 
  doc_feature_rdd = distinct_cids.map(get_doc_feature).filter(lambda x : x != None)

  os.system("hadoop fs -rm -r -skipTrash /user/work/zhangdongyue/qukan_log_stat/wlx_ct/{}/{}".format(thedate, thehour))
  distinct_cids.repartition(1).saveAsTextFile("/user/work/zhangdongyue/qukan_log_stat/wlx_ct/{}/{}".format(thedate, thehour))

  show_rdd = qukan_log_100_rdd.flatMap(flat_parse_show).filter(lambda x : x != None)

  ## 文章feature
  # content_fea_sql = '''
  #   select 
  #     id,
  #     type as cate,
  #     cover_show_type,
  #     content_type,
  #     subcategory,
  #     source_name,
  #     video_play_time,
  #     keywords_v2 as keywords,
  #     topic
  #   from 
  #     algo_qukan.qukan_rec_content_feature
  #   where 
  #     ds = '%s'
  # ''' % thedate
  #content_fea_table_rdd = spark.sql(content_fea_sql).rdd.map(content_fea_format).filter(lambda x : x != None)
  
  ## 提取用户基本信息 qttdc.usp_user_profile
  # user_static_profile_sql = '''
  #   select 
  #     memeber_id as user_id,
  #     last_login_device as device,
  #     reg_prov as prov,
  #     reg_city as city,
  #     sex as gender,
  #     last_login_client as phone_client,
  #     case when wlx_package_code is not NULL then wlx_package_code
  #          else nlx_source_code
  #     end as dtu,
  #     install_app_list as app_list
  #   from 
  #     qttdc.usp_user_profile
  #   where
  #     day = '%s'
  # ''' % thedate  
  #user_static_profile_rdd = spark.sql(user_static_profile_sql).rdd
  #user_info_rdd = user_static_profile_rdd.map(format_user_static_fea).filter(lambda x: x != None)

  ## 提取点击日志
  qukan_log_sql_300 = '''
    select  
        c.user_id as user_id,
        c.device as device,
        c.content_id as content_id,
        case when d.read_time is null then 0.0
             else d.read_time 
        end as read_time
    from
    (
      select 
        a.cookie_id as user_id,
        a.device as device,
        a.content_id as content_id
      from
      (
        select 
          cookie_id,
          device,
          field['content_id'].string_type as content_id
        from 
          bdm.qukan_log_cmd_p_byhour
        where
          day = '%s' and
          hour = '%s' and
          cmd = '300' and
          field['channel'].string_type = '255' and 
          field['from'].string_type = '11' and
          field['is_appview'].string_type <> '2'
      ) a
        left outer join
      (
       select 
        device_id 
       from 
        algo_qukan.gdt_new_user_contentid_info_1d_d
       where 
        content_id > 0
      ) b
      on 
        a.device = b.device_id
      where 
        b.device_id is not NULL
    ) c 
      left outer join
    (
      select 
        max(case when header_mi is null then device
             else header_mi
        end) as user_id,
        header_mi,
        device,
        max(cast(field['useTime'].string_type as double)) as read_time,
        field['selectedId'].string_type as content_id
      from
        bdm.qukan_client_collect_cmd_p_byhour
      where
        day = '%s' and
        cmd = '4034' and
        field['channel'].string_type = '255' and
        field['selectedId'].string_type is not null and
        device is not null and device <> '' and device <> '0'
      group by
        device, header_mi, field['selectedId'].string_type
    ) d
    on
      c.user_id = d.user_id and 
      c.content_id = d.content_id
  ''' % (thedate, thehour, thedate)

  qukan_log_300_rdd = spark.sql(qukan_log_sql_300).rdd
  click_rdd = qukan_log_300_rdd.map(parse_click).filter(lambda x : x != None)

  stat_rdd = show_rdd.leftOuterJoin(click_rdd).map(tuple_format)

  merge_rdd = stat_rdd.leftOuterJoin(doc_feature_rdd).map(merge_user_item_feature).filter(lambda x : x != None)

  os.system("hadoop fs -rm -r -skipTrash /user/work/zhangdongyue/qukan_log_stat/wlx_sample/{}/{}".format(thedate, thehour))
  merge_rdd.repartition(100).saveAsTextFile("/user/work/zhangdongyue/qukan_log_stat/wlx_sample/{}/{}".format(thedate, thehour))
  #device_id_info.write.saveAsTable("algo_qukan.gdt_new_user_dev_id_info_d")

  hsql = ''' 
    alter table algo_qukan.qukan_user_merge_log_h 
    add if not exists partition (ds='%s', hh='%s')
    location 'hdfs:///user/work/zhangdongyue/qukan_log_stat/wlx_sample/%s/%s'
  ''' % (thedate, thehour, thedate, thehour)

  spark.sql(hsql)

if __name__ == '__main__':
  merge_log()
  httpClient.close()
  httpDocFeaClient.close()

## ~
