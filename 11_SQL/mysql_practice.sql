
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

-- ���� ������ ����

select * from fast.visit;
select count(customer_id) from fast.visit;

select 
	v2.visited_month,
	count(v.customer_id)
from 
	fast.visit v 
	inner join (select
					customer_id,
					date_format(visited_at, '%Y-%m') as visited_month
				from 
					fast.visit) v2
	on v.customer_id = v2.customer_id
group by
	1;

select 
	*
from 
	fast.visit v 
	left join (select
					customer_id,
					date_format(visited_at, '%m') as visited_month
				from 
					fast.visit) v2
	on v.customer_id = v2.customer_id;


-- 4. 7���� ���� ������ �� ��� ���űݾ��� ��� �ǳ���?

select * from fast.purchase p ;

select
	avg(db.revenue)
from
	(select
		customer_id,
		sum(price) as revenue
	from
		fast.purchase p 
	where 
		purchased_at >= '2020-07-01 00:00:00'
	and purchased_at < '2020-08-01 00:00:00'
	group by 
		1) as db


-- 5. 7���� ���� ���� ������ �� Top3�� Top10~15 ���� �̾��ּ���
		
select * from fast.purchase;

select 
	customer_id,
	count(id)
from 
	fast.purchase p 
where 
	purchased_at >= '2020-07-01 00:00:00'
and purchased_at < '2020-08-01 00:00:00'
group by
	1
order by
	2 desc
limit 3;
	
select 
	customer_id,
	count(id)
from 
	fast.purchase p 
where 
	purchased_at >= '2020-07-01 00:00:00'
and purchased_at < '2020-08-01 00:00:00'
group by
	1
order by
	2 desc
limit
	5 offset 10;
	

-- 6. 2020�� 7���� ��� DAU�� �����ּ���, Activity User ���� �߼� �����ϴ� �߼��ΰ���?
select * from fast.visit v ;

select
 	date_format(visited_at, '%Y-%m-%d') as date_at,
 	count(customer_id)
from 
	fast.visit v
where 
	visited_at >= '2020-07-01 00:00:00'
and visited_at < '2020-08-01 00:00:00'
group by 
	1
;

	
 -- 7. 2020�� 7���� ��� WAU�� �����ּ���.

select * from fast.visit v ;

select 
	round(avg(cnt), 0)
from
	(select 
		date_format(visited_at, '%Y-%m-%U') as date_at,
		count(distinct customer_id) as cnt
	from 
		fast.visit
	where 
		visited_at >= '2020-07-05 00:00:00'
	and visited_at < '2020-07-26 00:00:00'
	group by 
		1) as foo;
		
		

select 
	date_format(visited_at, '%Y-%m-%U') as date_at,
	count(distinct customer_id) as cnt
from 
	fast.visit
where 
	visited_at >= '2020-07-05 00:00:00'
and visited_at < '2020-07-26 00:00:00'
group by 
	1;
	

select 
	customer_id,
	visited_at,
	date_format(visited_at, '%Y-%m-%U') as date_at
from 
	fast.visit
where 
	visited_at >= '2020-07-05 00:00:00'
and visited_at < '2020-07-26 00:00:00';

select 
	round(avg(foo.cnt),0)
from(
	select 
		date_format(visited_at, '%Y-%m-%U') as date_at,
		count(distinct customer_id) as cnt
	from
		fast.visit 
	where 
		visited_at >= '2020-07-05 00:00:00'
	and visited_at < '2020-07-26 00:00:00'
	group by 
		1) as foo;

-- 8. 2020�� 7�� 7���� Daily Revenue�� �����ϴ� �߼��ΰ���? ��� Daily Revenue�� �����ּ���.

select * from fast.purchase;	

select 
	date_format(purchased_at, '%Y-%m-%d'),
	sum(price)
from 
	fast.purchase p 
where 
	purchased_at >= '2020-07-01 00:00:00'
and purchased_at < '2020-08-01 00:00:00'
group by
	1;

select 
	round(avg(foo.sum_price),0)
from(
select 
	date_format(purchased_at, '%Y-%m-%d') as date_at,
	sum(price) as sum_price
from 
	fast.purchase p 
where 
	purchased_at >= '2020-07-01 00:00:00'
and purchased_at < '2020-08-01 00:00:00'
group by
	1) as foo
	

-- 9. 2020�� 7���� ��� Weekly Revenue�� �����ּ���
	
select * from fast.purchase;

select 
	round(avg(price_sum),0)
from(
	select 
		date_format(purchased_at, '%Y-%m-%U') as data_at,
		sum(price) as price_sum
	from 
		fast.purchase p 
	where 
		purchased_at >= '2020-07-05 00:00:00'
	and purchased_at < '2020-07-26 00:00:00'
	group by 
		1) as foo

-- 10. 2020�� 7�� ���Ϻ� Revenue�� �����ּ���. ��� ������ Revenue�� ���� ���� ��� ������ Revenue�� ���� ������?
		
select * from fast.purchase p ;

select 
	date_format(purchased_at, '%W'),
	sum(price)
from 
	fast.purchase p 
where 
	purchased_at >= '2020-07-05 00:00:00'
and purchased_at < '2020-07-26 00:00:00'
group by 
	1;
	

select 
	date_format(date_at, '%w') as day_of_week,
	date_format(date_at, '%W') as day_name,
	avg(revenue) as daily_revenue
from 
	(select 
		date_format(purchased_at, '%Y-%m-%d') as date_at,
		sum(price) as revenue
	from 
		fast.purchase p 
	where 
		purchased_at >= '2020-07-01 00:00:00'
	and purchased_at < '2020-08-01 00:00:00'
	group by 
		1) foo
group by 
	1, 2
order by 
	1;
	

-- 11. 2020�� 7�� �ð��뺰 Revenue�� �����ּ���. ��� �ð��밡 Revenue�� ���� ���� ��� �Ⱓ�밡 Revenue�� ���� ������?
select * from fast.purchase p ;

select 
	date_format(purchased_at, '%Y-%m-%d') as date_at,
	date_format(purchased_at, '%H') as hour_at,
	sum(price) as sum_price
from 
	fast.purchase p 
where 
	purchased_at >= '2020-07-01 00:00:00'
and purchased_at < '2020-08-01 00:00:00'
group by
	1, 2;


select 
	hour_at,
	round(avg(sum_price),0) as avg_price
from(
	select 
		date_format(purchased_at, '%Y-%m-%d') as date_at,
		date_format(purchased_at, '%H') as hour_at,
		sum(price) as sum_price
	from 
		fast.purchase p 
	where 
		purchased_at >= '2020-07-01 00:00:00'
	and purchased_at < '2020-08-01 00:00:00'
	group by
		1, 2) as foo
group by 
	1
order by 
	2 desc;

-- 12. 2020�� 7�� ���� �� �ð��뺰 Revenue�� �����ּ���. ��� ���� �� �ð��밡 Revenue�� ���� ���� ��� �ð��밡 Revenue�� ���� ������?
select * from fast.purchase p ;

select 
	date_format(purchased_at, '%Y-%m-%d') as date_at,
 	date_format(purchased_at, '%W') as week_at,
    date_format(purchased_at, '%H')as hour_at,
	sum(price)
from 
	fast.purchase p  
where 
	purchased_at >= '2020-07-01 00:00:00'
and purchased_at < '2020-08-01 00:00:00'
group by 
	1,2,3; 
	
select 
	week_at,
	hour_at,
	avg(revenue)
from (
	select 
		date_format(purchased_at, '%Y-%m-%d') as date_at,
	 	date_format(purchased_at, '%W') as week_at,
	    date_format(purchased_at, '%H')as hour_at,
		sum(price) as revenue
	from 
		fast.purchase p  
	where 
		purchased_at >= '2020-07-01 00:00:00'
	and purchased_at < '2020-08-01 00:00:00'
	group by 
		1,2,3) as foo
group by 
 1,2
order by 
3;


-- ���� �� �ð��� �� Activity User �� ���
select * from fast.visit v ;

select 
	week_at,
	hour_at,
	avg(customer_count)
from(select 
	date_format(visited_at, '%Y-%m-%d') as date_at,
	date_format(visited_at, '%W') as week_at,
	date_format(visited_at, '%H') as hour_at,
	count(distinct customer_id) as customer_count
from 
	fast.visit v 
where 
	visited_at >= '2020-07-01 00:00:00'
and visited_at < '2020-08-01 00:00:00'
group by
	1,2,3) as foo 
group by
	1,2;


-- 13. ��ü ������ Demographic�� �˰� �;��. ��, ���ɺ��� ���� ���ڸ� �˷��ּ���.
select * from fast.customer c ;

select
	case
		when length(gender) < 1 then 'O'
		when gender is null then 'O'
		else gender
	end as gender,
	case 
		when age is null then '������'
		when age <= 5 then '0~5����'
		when age <= 10 then '6�̻�~10����'
		when age <= 15 then '11�̻�~15����'
		when age <= 20 then '16�̻�~20����'
		when age <= 25 then '21�̻�~25����'
		when age <= 30 then '25�̻�~30����'
		when age <= 35 then '31�̻�~35����'
		when age <= 40 then '36�̻�~40����'
		when age <= 45 then '41�̻�~45����'
		when age >= 46 then '46�̻�'
	end as age,
	age,
	count(distinct customer_id)
from 
	fast.customer c 
group by 
	1,2
	
-- 14. 13 ����� ��,������ ����(����)(ex.����(25-29�� ����)) ���� �������ֽð�, �� ��, ������ ��ü ������ �󸶳� �����ϴ��� ����(%)�� �˷��ּ���, ���� ������ ���� ������� �˷��ּ���.
select
	concat(
	case
		when gender = 'M' then '����'
		when gender = 'F' then '����'
		when gender = 'O' then '��Ÿ'
		when length(gender) < 1 then '��Ÿ'
		when gender is null then '��Ÿ'
		else gender
	end,
	'(',
	case 
		when age is null then '������'
		when age <= 5 then '0~5����'
		when age <= 10 then '6�̻�~10����'
		when age <= 15 then '11�̻�~15����'
		when age <= 20 then '16�̻�~20����'
		when age <= 25 then '21�̻�~25����'
		when age <= 30 then '25�̻�~30����'
		when age <= 35 then '31�̻�~35����'
		when age <= 40 then '36�̻�~40����'
		when age <= 45 then '41�̻�~45����'
		when age >= 46 then '46�̻�'
	end,
	')') as gen_age,
	count(distinct customer_id),
	round(count(distinct customer_id)/(select count(distinct customer_id) from fast.customer c2)*100, 2) as per
from 
	fast.customer c 
group by 
	1
order by 
	3 desc;

-- 15. 2020�� 7���� ������ ���� ���� �Ǽ���, �� Revenue�� �����ּ���. ���� �̿��� ������ �ϳ��� �����ּ���.
select * from fast.purchase p ;

select 
	case 
		when c.gender not in ('F', 'M') then '��Ÿ'
		else c.gender 
	end as gender,
	count(distinct p.customer_id),
	sum(price)
from 
	fast.purchase p 
	left join fast.customer c on p.customer_id = c.customer_id 
where 
	p.purchased_at >= '2020-07-01 00:00:00'
and p.purchased_at < '2020-08-01 00:00:00'
group by
	1;


-- 16. 2020�� 7���� ����/���ɴ뿡 ���� ���ŰǼ���, �� Revenue�� �����ּ���. ���� �̿��� ������ �ϳ��� �����ּ���.
select * from fast.customer c ;

select 
	case 
		when c.gender not in ('F', 'M') then '��Ÿ'
		else c.gender 
	end as gender,
	case 
		when age is null then '������'
		when age <= 5 then '0~5����'
		when age <= 10 then '6�̻�~10����'
		when age <= 15 then '11�̻�~15����'
		when age <= 20 then '16�̻�~20����'
		when age <= 25 then '21�̻�~25����'
		when age <= 30 then '25�̻�~30����'
		when age <= 35 then '31�̻�~35����'
		when age <= 40 then '36�̻�~40����'
		when age <= 45 then '41�̻�~45����'
		when age >= 46 then '46�̻�'
	end as age,
	count(distinct p.customer_id) as cnt,
	sum(p.price) as sum_price
from 
	fast.purchase p 
	left join fast.customer c on p.customer_id = c.customer_id 
where 
	p.purchased_at >= '2020-07-01 00:00:00'
and p.purchased_at < '2020-08-01 00:00:00'
group by
	1,2;
	

-- 17. 2020�� 7�� �Ϻ� ����� ������, �������� �����ּ���.
select * from fast.purchase p ;

select
	*,
	lag(sum_price) over(),
	round((sum_price - lag(sum_price) over())/lag(sum_price) over()*100,2)
from (
	select 
		date_format(purchased_at, '%Y-%m-%d') as date_dt,
		sum(price) as sum_price
	from 
		fast.purchase p 
	where 
		p.purchased_at >= '2020-07-01 00:00:00'
	and p.purchased_at < '2020-08-01 00:00:00'
	group by 
		1) as foo;
	

-- 18. 2020�� 7�� �Ϻ��� ���� ������ �������� ������ ������ �ٷ����ؿ�. 7���� �Ϻ��� ���� �ݾ� �������� ���� ���� ������ �� top3�� �̾��ּ���
select * from fast.purchase p ;

select 
	*
from(
select 
	date_format(purchased_at, '%Y-%m-%d'),
	customer_id,
	sum(price),
	dense_rank() over(partition by date_format(purchased_at, '%Y-%m-%d') order by sum(price) desc) as rank_rev
from 
	fast.purchase p 
where 
	purchased_at >= '2020-07-01 00:00:00'
and purchased_at < '2020-08-01 00:00:00'
group by
	1,2) as foo
where
	rank_rev < 4;
	

-- 19. 2020�� 7�� �츮 �ű������� �Ϸ� �ȿ� ������ �Ѿ�� ������ ��� �ǳ���? �� ������ ��� �˰�Ͱ�, �������� ���� �� �� ������ �ҿ�Ǵ��� �˰�;��.
select * from fast.purchase p ;
select * from fast.visit v ;


select
	customer_id,
	min(purchased_at) as purchased_at
from
	fast.purchase p
group by
	1;



select
	a.*,
	b.customer_id as paying_user,
	b.purchased_at,
	time_to_sec(timediff(b.purchased_at, a.created_at))/3600 as diff_hours
from fast.customer a
	left join (select
					customer_id,
					min(purchased_at) as purchased_at
				from
					fast.purchase p
				group by
					1) b on a.customer_id = b.customer_id 
			and b.purchased_at < a.created_at + interval 1 day
where 
	a.created_at >= '2020-07-01'
and a.created_at < '2020-08-01';


with tb1 as (
select
	a.*,
	b.customer_id as paying_user,
	b.purchased_at,
	time_to_sec(timediff(b.purchased_at, a.created_at))/3600 as diff_hours
from fast.customer a
	left join (select
					customer_id,
					min(purchased_at) as purchased_at
				from
					fast.purchase p
				group by
					1) b on a.customer_id = b.customer_id 
			and b.purchased_at < a.created_at + interval 1 day
where 
	a.created_at >= '2020-07-01'
and a.created_at < '2020-08-01')
select
	round(count(paying_user)/count(customer_id)*100,2)
from 
	tb1 
union all
select 
	round(avg(diff_hours), 2)
from 
	tb1;


select 
	a.*,
	b.customer_id,
	b.min_purchased_at
from 
	fast.customer a
	left join(
	select 
		customer_id,
		min(purchased_at) as min_purchased_at
	from 
		fast.purchase
	group by
	1) b on a.customer_id = b.customer_id 
	and a.created_at + interval 1 day > b.purchased_at
	;
	
	
-- �ѹ� �� 19. 2020�� 7�� �츮 �ű������� �Ϸ� �ȿ� ������ �Ѿ�� ������ ��� �ǳ���? �� ������ ��� �˰�Ͱ�, �������� ���� �� �� ������ �ҿ�Ǵ��� �˰�;��.

select * from fast.customer;
select * from fast.purchase;


with tb1 as (
select 
	a.*,
	b.customer_id as paying_user,
	b.min_purchased_at,
	time_to_sec(timediff(b.min_purchased_at, a.created_at))/3600 as diff_hours
from 
	fast.customer a
	left join (
		select 
			customer_id,
			min(purchased_at) as min_purchased_at
		from 
			fast.purchase p 
		group by 
			1) b
		on a.customer_id = b.customer_id
		and a.created_at + interval 1 day > b.min_purchased_at
where 
	a.created_at >= '2020-07-01'
and a.created_at < '2020-08-01')
select 
	round(count(paying_user)/count(customer_id),2)
from
	tb1
union all
select 
	round(avg(diff_hours),2)
from 
	tb1;
	



select 
	date_format(purchased_at, '%Y-%m-%d'),
	count(customer_id)
from 
	fast.purchase p 
group by 
	1
	
