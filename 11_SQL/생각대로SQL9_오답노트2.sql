-- 문제 3-1
with tb1 as 
(
select 
	'G' as rating,
	'General Audiences' as eng_text,
	'(모든 연령대 시청가능)' as kor_text
union all
select
	'PG' as rating,
	'Parental Guidance Suggested' as eng_text,
	'(모든 연령대 시청가능하나, 부모의 지도가 필요)' as kor_text
union all
select
	'PG-13' as rating,
	'Parents Strongly Cautioned'as eng_text,
	'(13세 미만의 아동에게 부적절 할 수 있으며, 부모의 주의를 요함)' as kor_text
union all
select
	'R' as rating,
	'Restricted' as eng_text,
	'(17세 또는 그이상의 성인)' as kor_text
union all
select
	'NC-17' as rating,
	'No One 17 and Under Admitted' as eng_text,
	'(17세 이하 시청 불가)' as kor_text
)
select *
from tb1;


-- 문제 3-2
select * from film limit 10;

with tb1 as 
(
select 
	'G' as rating,
	'General Audiences' as eng_text,
	'(모든 연령대 시청가능)' as kor_text
union all
select
	'PG' as rating,
	'Parental Guidance Suggested' as eng_text,
	'(모든 연령대 시청가능하나, 부모의 지도가 필요)' as kor_text
union all
select
	'PG-13' as rating,
	'Parents Strongly Cautioned'as eng_text,
	'(13세 미만의 아동에게 부적절 할 수 있으며, 부모의 주의를 요함)' as kor_text
union all
select
	'R' as rating,
	'Restricted' as eng_text,
	'(17세 또는 그이상의 성인)' as kor_text
union all
select
	'NC-17' as rating,
	'No One 17 and Under Admitted' as eng_text,
	'(17세 이하 시청 불가)' as kor_text
)
select
	f.film_id,
	f.rating,
	tb1.eng_text,
	tb1.kor_text
from film f
	left outer join tb1 on cast(f.rating as varchar) = tb1.rating;
	

-- 문제 5번
with tb1 as
(
select 
	'A' as chk1,
	'A가 처음' as flag
union all
select
	'B' as chk1,
	'B가 처음' as flag
union all
select
	'C' as chk1,
	'C가 처음' as flag
)
select
	customer_id,
	first_name,
	tb1.chk1,
	tb1.flag,
	coalesce(tb1.flag, 'Other')
from 
	customer c
	left outer join tb1 on substring(c.first_name,1,1) = tb1.chk1;
	

-- 문제 8-1
select 
	min(return_date),
	max(return_date)
from
	rental r;
	

-- 문제 8-2
with tb1 as
(
select 
	cast('2005-01-01 00:00:00' as timestamp) as chk1,
	cast('2005-03-31 23:59:59' as timestamp) as chk2,
	'Q1' as quater
union all
select 
	cast('2005-04-01 00:00:00' as timestamp) as chk1,
	cast('2005-06-30 23:59:59' as timestamp) as chk2,
	'Q2' as quater
union all
select 
	cast('2005-07-01 00:00:00' as timestamp) as chk1,
	cast('2005-09-30 23:59:59' as timestamp) as chk2,
	'Q3' as quater
union all
select 
	cast('2005-01-01 00:00:00' as timestamp) as chk1,
	cast('2005-03-31 23:59:59' as timestamp) as chk2,
	'Q4' as quater
)
select
	r.*,
	tb1.quater
from
	rental r
	left outer join tb1 on r.return_date  between tb1.chk1 and tb1.chk2;


--문제 10
with new_password as
(
select
	1 as staff_id,
	'12345' as new_pwd
union all
select
	2 as staff_id,
	'54321' as new_pwd
)
select
	s.staff_id,
	s.password as origin_pwd,
	np.new_pwd
from
	staff s
	join new_password as np on s.staff_id = np.staff_id;