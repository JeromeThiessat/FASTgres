SELECT COUNT(*)
FROM
tag as t,
site as s,
question as q,
tag_question as tq
WHERE
t.site_id = s.site_id
AND q.site_id = s.site_id
AND tq.site_id = s.site_id
AND tq.question_id = q.id
AND tq.tag_id = t.id
AND (s.site_name in ('askubuntu'))
AND (t.name in ('10.04','12.10','apache2','firefox','kernel','kubuntu','partitioning','scripts','shortcut-keys','system-installation','updates','virtualbox'))
AND (q.score >= 1)
AND (q.score <= 10)
