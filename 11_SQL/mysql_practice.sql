
-- 1. 2020년 7월의 총 Revenue를 구해주세요

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


-- 2. 2020년 7월의 MAU를 구해주세요

select 
	count(distinct customer_id)
from
	fast.visit
where 
	date(visited_at) >= '2020-07-01'
and date(visited_at) < '2020-08-01';


-- 3. 7월에 우리 Active 유저의 구메율(Paying Rate)은 어떻게 되나요?

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

-- 월별 구매율 보기

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


-- 4. 7월에 구매 유저의 월 평균 구매금액은 어떻게 되나요?

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


-- 5. 7월에 가장 많이 구매한 고객 Top3와 Top10~15 고객을 뽑아주세요
		
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
	

-- 6. 2020년 7월의 평균 DAU를 구해주세요, Activity User 수가 추세 증가하는 추세인가요?
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

	
 -- 7. 2020년 7월의 평균 WAU를 구해주세요.

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

-- 8. 2020년 7월 7월의 Daily Revenue는 증가하는 추세인가요? 평균 Daily Revenue도 구해주세요.

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