-- ���� 3-1
with tb1 as 
(
select 
	'G' as rating,
	'General Audiences' as eng_text,
	'(��� ���ɴ� ��û����)' as kor_text
union all
select
	'PG' as rating,
	'Parental Guidance Suggested' as eng_text,
	'(��� ���ɴ� ��û�����ϳ�, �θ��� ������ �ʿ�)' as kor_text
union all
select
	'PG-13' as rating,
	'Parents Strongly Cautioned'as eng_text,
	'(13�� �̸��� �Ƶ����� ������ �� �� ������, �θ��� ���Ǹ� ����)' as kor_text
union all
select
	'R' as rating,
	'Restricted' as eng_text,
	'(17�� �Ǵ� ���̻��� ����)' as kor_text
union all
select
	'NC-17' as rating,
	'No One 17 and Under Admitted' as eng_text,
	'(17�� ���� ��û �Ұ�)' as kor_text
)
select *
from tb1;


-- ���� 3-2
select * from film limit 10;

with tb1 as 
(
select 
	'G' as rating,
	'General Audiences' as eng_text,
	'(��� ���ɴ� ��û����)' as kor_text
union all
select
	'PG' as rating,
	'Parental Guidance Suggested' as eng_text,
	'(��� ���ɴ� ��û�����ϳ�, �θ��� ������ �ʿ�)' as kor_text
union all
select
	'PG-13' as rating,
	'Parents Strongly Cautioned'as eng_text,
	'(13�� �̸��� �Ƶ����� ������ �� �� ������, �θ��� ���Ǹ� ����)' as kor_text
union all
select
	'R' as rating,
	'Restricted' as eng_text,
	'(17�� �Ǵ� ���̻��� ����)' as kor_text
union all
select
	'NC-17' as rating,
	'No One 17 and Under Admitted' as eng_text,
	'(17�� ���� ��û �Ұ�)' as kor_text
)
select
	f.film_id,
	f.rating,
	tb1.eng_text,
	tb1.kor_text
from film f
	left outer join tb1 on cast(f.rating as varchar) = tb1.rating;
	

-- ���� 5��
with tb1 as
(
select 
	'A' as chk1,
	'A�� ó��' as flag
union all
select
	'B' as chk1,
	'B�� ó��' as flag
union all
select
	'C' as chk1,
	'C�� ó��' as flag
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
	

-- ���� 8-1
select 
	min(return_date),
	max(return_date)
from
	rental r;
	

-- ���� 8-2
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


--���� 10
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