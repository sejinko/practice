
-- 1. 2020�� 7���� �� Revenue�� �����ּ���

select * from fast.customer c ;
select * from fast.purchase p ;
select * from fast.visit v ;

select
	sum(price)
from 
	fast.purchase
where
	date(purchased_at) >= '2020-07-01'
and date(purchased_at) < '2020-08-01'
;


-- 2. 2020�� 7���� MAU�� �����ּ���

select 
	count(distinct customer_id)
from
	fast.visit
where 
	date(visited_at) >= '2020-07-01'
and date(visited_at) < '2020-08-01';


-- 3. 7���� �츮 Active ������ ������(Paying Rate)�� ��� �ǳ���?

select 
	count(distinct customer_id)
from 
	fast.visit v 
where 
	visited_at >= '2020-07-01 00:00:00'
and visited_at < '2020-08-01 00:00:00';



select 
	count(distinct customer_id)
from
	fast.purchase p 
where 
	purchased_at >= '2020-07-01 00:00:00'
and purchased_at < '2020-08-01 00:00:00';


select round((11174/16414)*100,2);

-- 