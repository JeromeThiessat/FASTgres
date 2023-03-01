
select count(distinct q1.id) from
site TABLESAMPLE SYSTEM (60),
 post_link pl1 TABLESAMPLE SYSTEM (60), 
 post_link pl2 TABLESAMPLE SYSTEM (60), 
 question q1 TABLESAMPLE SYSTEM (60), 
 question q2 TABLESAMPLE SYSTEM (60), 
 question q3 TABLESAMPLE SYSTEM (60) 
 
where
site.site_name = 'eosio' and
q1.site_id = site.site_id and
q1.site_id = q2.site_id and
q2.site_id = q3.site_id and

pl1.site_id = q1.site_id and
pl1.post_id_from = q1.id and
pl1.post_id_to = q2.id and

pl2.site_id = q1.site_id and
pl2.post_id_from = q2.id and
pl2.post_id_to = q3.id and

exists ( select * from comment where comment.site_id = q3.site_id and comment.post_id = q3.id ) and
exists ( select * from comment where comment.site_id = q2.site_id and comment.post_id = q2.id ) and
exists ( select * from comment where comment.site_id = q1.site_id and comment.post_id = q1.id ) and

q1.score > q3.score;
