-- 조회와 필터링
-- 실습 1
select 
	distinct a.customer_id,
	a.amount
from
	payment a
where a.amount =
	(
		select
			b.amount
		from
			payment b
		order by
			b.amount desc
		limit 1
	)
order by a.customer_id asc;
	
-- 실습 2
select 
	a.customer_id,
	a.email
from
	customer a
where 
	a.email not like '%@'
	and 
	a.email not like '@%'
	and 
	a.email like '%@%';


-- 조인과 집계
-- 실습 1
select 
	to_char(a.rental_date, 'yyyy'),
	count(a.rental_id)
from
	rental as a
group by
	to_char(a.rental_date, 'yyyy');
	

select 
	to_char(a.rental_date, 'yyyymm'),
	count(a.rental_id)
from
	rental as a
group by
	to_char(a.rental_date, 'yyyymm');


select 
	to_char(a.rental_date, 'yyyymmdd') date,
	count(a.rental_id)
from
	rental a
group by
	to_char(a.rental_date, 'yyyymmdd')
order by 
	date;


select 
	to_char(a.rental_date, 'yyyy'),
	to_char(a.rental_date, 'mm'),
	to_char(a.rental_date, 'dd'),
	count(a.rental_id)
from
	rental a
group by
rollup 
( 
	to_char(a.rental_date, 'yyyy'),to
	to_char(a.rental_date, 'mm'),
	to_char(a.rental_date, 'dd')
);


select 
	to_char(a.rental_date, 'yyyy') as year,
	to_char(a.rental_date, 'mm') as month,
	to_char(a.rental_date, 'dd') as day,
	count(a.rental_id)
from
	rental a
group by
grouping sets 
(
	(
		to_char(a.rental_date, 'yyyy'),
		to_char(a.rental_date, 'mm'),
		to_char(a.rental_date, 'dd')
	),
	(
		to_char(a.rental_date, 'yyyy'),
		to_char(a.rental_date, 'mm')
	),
	(
		to_char(a.rental_date, 'yyyy')
	),
	(
	)
);
	
	


select 
	to_char(a.rental_date, 'yyyy') as year,
	to_char(a.rental_date, 'mm') as month,
	to_char(a.rental_date, 'dd') as day,
	count(a.rental_id)
from
	rental a
group by
	to_char(a.rental_date, 'yyyy'),
	to_char(a.rental_date, 'mm'),
	to_char(a.rental_date, 'dd')
union all 
select 
	to_char(a.rental_date, 'yyyy'),
	to_char(a.rental_date, 'mm'),
	null,
	count(a.rental_id)
from
	rental a
group by
	to_char(a.rental_date, 'yyyy'),
	to_char(a.rental_date, 'mm')
union all 
select 
	to_char(a.rental_date, 'yyyy'),
	null,
	null,
	count(a.rental_id)
from
	rental a
group by
	to_char(a.rental_date, 'yyyy')
union all 
select 
	null,
	null,
	null,
	count(a.rental_id)
from
	rental a
order by 
	year asc,
	month asc,
	day asc;


-- 실습 2
select 
	customer_id,
	count(rental_id)
from
	rental
group by
	customer_id
order by
	customer_id;



select 
	a.customer_id,
	count(a.rental_id),
	rank() over(order by count(a.rental_id) desc),
	dense_rank() over(order by count(a.rental_id) desc),
	row_number() over(order by count(a.rental_id) desc),
	max(first_name),
	min(last_name)
from 
	rental a,
	customer b
where
	a.customer_id = b.customer_id	
group by
	a.customer_id
limit 
	1;
