
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
	

-- 9. 2020년 7월의 평균 Weekly Revenue를 구해주세요
	
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

-- 10. 2020년 7월 요일별 Revenue를 구해주세요. 어느 요일이 Revenue가 가장 높고 어느 요일이 Revenue가 가장 낮나요?
		
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
	

-- 11. 2020년 7월 시간대별 Revenue를 구해주세요. 어느 시간대가 Revenue가 가장 높고 어느 기간대가 Revenue가 가장 낮나요?
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

-- 12. 2020년 7월 요일 및 시간대별 Revenue를 구해주세요. 어느 요일 및 시간대가 Revenue가 가장 높고 어느 시간대가 Revenue가 가장 낮나요?
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


-- 요일 및 시간대 별 Activity User 수 계산
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


-- 13. 전체 유저의 Demographic을 알고 싶어요. 성, 연령별로 유저 숫자를 알려주세요.
select * from fast.customer c ;

select
	case
		when length(gender) < 1 then 'O'
		when gender is null then 'O'
		else gender
	end as gender,
	case 
		when age is null then '무응답'
		when age <= 5 then '0~5이하'
		when age <= 10 then '6이상~10이하'
		when age <= 15 then '11이상~15이하'
		when age <= 20 then '16이상~20이하'
		when age <= 25 then '21이상~25이하'
		when age <= 30 then '25이상~30이하'
		when age <= 35 then '31이상~35이하'
		when age <= 40 then '36이상~40이하'
		when age <= 45 then '41이상~45이하'
		when age >= 46 then '46이상'
	end as age,
	age,
	count(distinct customer_id)
from 
	fast.customer c 
group by 
	1,2
	
-- 14. 13 결과의 성,연령을 성별(연령)(ex.남성(25-29세 이하)) 으로 통합해주시고, 각 성, 연령이 전체 고객에서 얼마나 차지하는지 분포(%)를 알려주세요, 역시 분포가 높은 순서대로 알려주세요.
select
	concat(
	case
		when gender = 'M' then '남성'
		when gender = 'F' then '여성'
		when gender = 'O' then '기타'
		when length(gender) < 1 then '기타'
		when gender is null then '기타'
		else gender
	end,
	'(',
	case 
		when age is null then '무응답'
		when age <= 5 then '0~5이하'
		when age <= 10 then '6이상~10이하'
		when age <= 15 then '11이상~15이하'
		when age <= 20 then '16이상~20이하'
		when age <= 25 then '21이상~25이하'
		when age <= 30 then '25이상~30이하'
		when age <= 35 then '31이상~35이하'
		when age <= 40 then '36이상~40이하'
		when age <= 45 then '41이상~45이하'
		when age >= 46 then '46이상'
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

-- 15. 2020년 7월의 성별에 따라 구매 건수와, 총 Revenue를 구해주세요. 남녀 이외의 성별은 하나로 묶어주세요.
select * from fast.purchase p ;

select 
	case 
		when c.gender not in ('F', 'M') then '기타'
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


-- 16. 2020년 7월의 성별/연령대에 따라 구매건수와, 총 Revenue를 구해주세요. 남녀 이외의 성별은 하나로 묶어주세요.
select * from fast.customer c ;

select 
	case 
		when c.gender not in ('F', 'M') then '기타'
		else c.gender 
	end as gender,
	case 
		when age is null then '무응답'
		when age <= 5 then '0~5이하'
		when age <= 10 then '6이상~10이하'
		when age <= 15 then '11이상~15이하'
		when age <= 20 then '16이상~20이하'
		when age <= 25 then '21이상~25이하'
		when age <= 30 then '25이상~30이하'
		when age <= 35 then '31이상~35이하'
		when age <= 40 then '36이상~40이하'
		when age <= 45 then '41이상~45이하'
		when age >= 46 then '46이상'
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
	

-- 17. 2020년 7월 일별 매출과 증감폭, 증감률을 구해주세요.
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
	

-- 18. 2020년 7월 일별로 많이 구매한 고객들한테 소정의 선물을 줄려고해요. 7월에 일별로 구매 금액 기준으로 가장 많이 지출한 고객 top3를 뽑아주세요
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
	

-- 19. 2020년 7월 우리 신규유저가 하루 안에 결제로 넘어가는 비율이 어떻게 되나요? 그 비율이 어떤지 알고싶고, 결제까지 보통 몇 분 정도가 소요되는지 알고싶어요.
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
	
	
-- 한번 더 19. 2020년 7월 우리 신규유저가 하루 안에 결제로 넘어가는 비율이 어떻게 되나요? 그 비율이 어떤지 알고싶고, 결제까지 보통 몇 분 정도가 소요되는지 알고싶어요.

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
	
