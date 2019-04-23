set mapreduce.job.queuename=root.rec.biz;
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
SET hive.exec.max.dynamic.partitions=100000;
SET hive.exec.max.dynamic.partitions.pernode=100000;
SET mapreduce.reduce.memory.mb=3809;
SET mapreduce.reduce.java.opts=-Xmx3428m;
SET mapreduce.map.memory.mb=3809;
SET mapreduce.map.java.opts=-Xmx3428m;

-- dl_cpc.cpc_user_installed_apps

drop table algo_qukan.qukan_user_merge_log_h;
create table if not exists algo_qukan.qukan_user_merge_log_h (
        u_id                 int    comment '用户id',
        u_device_id          string comment '设备号',
        u_content_id         bigint comment '文章ID',
        u_wlx_content_id     bigint comment '为0表示没有获取到contentid, 内容外拉新来源内容ID',
        u_age                int    comment '年龄-缺失比较多',
        u_gender             int    comment '1-男，2-女，5-未知',
        u_phone_client       string comment 'android or ios',
        u_province           string comment '省',
        u_city               string comment '市',
        u_dtu                string comment '新用户安装app来源,目前只有安卓用户有',
        u_applist            string comment '|分隔,目前只有安卓用户有',
        u_readsourcename     string comment '|分隔',
        u_regtime            int    comment '注册时间,精确到秒,时间戳,天级离散化(7天)',
        u_favocate5          string comment '逗号分隔,一级分类,数字',
        u_keywords_vread     string comment '斗号分隔, 如有搜索词, 合并搜索词',
        u_keyword_cates      string comment '<kv>~<cate>,...',
        u_type_readlist      string comment '用户阅读历史,|分隔,根据一级分类做过打散',
        u_subcates           string comment '二级分类,|分隔',
        u_topics             string comment 'topics',
        i_recall_from        int    comment '召回来源ID',
        i_keywords           string comment '文章关键词,|分隔',
        i_subcate            string comment '文章二级分类',
        i_topics             string comment '文章topic',
        i_cate               int    comment '文章一级分类',
        i_type               int    comment '文章类型 视频/图文/图集',
        i_source_name        string comment '文章来源',
        i_covershowtype      int    comment '大图三图',
        i_playtime           int    comment '播放时长',
        i_avg_readtime       int    comment '平均阅读时长',
        i_show               int    comment '展现次数',
        i_read               int    comment '播放次数',
        i_pos                int    comment '文章展现每刷位置',
        l_click              int    comment '是否点击',
        l_read_time          int    comment '播放时长(s)',
        l_share              int    comment '是否分享 0-未分享 1-分享',
        l_comment            int    comment '是否评论 0-未评论 1-评论',
        l_praise             int    comment '是否点赞 0-未点赞 1-点赞',
        r_type               int    comment '1-train 0-test'
    ) comment 'merge log, u_<xx> 用户特征, i_<xx> item特征, l_<xx> label'
partitioned by (ds string, hh string)
row format delimited fields terminated by ',' 
;

--stored as textfile
--location '/user/work/zhangdongyue/ranking/hh/'
