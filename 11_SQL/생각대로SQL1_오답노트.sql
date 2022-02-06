-- 문제7번
select * from rental limit 10;

select
	*,
	date(return_date) - date(rental_date) + 1 as sub
from
	rental
where
	date(return_date) - date(rental_date) + 1 >= 10;




  -- datetime() 함수 찾아보기
select 
	datetime(return_date)
from
	rental;



  -- datediff() 함수 찾아보기
select 
	*
from
	rental
where 
	datediff(month, return_date, rental_date) > 1;



  -- cast( as date) / datetime 찾아보기
select 
	cast(return_date as date)
from
	rental;
	


-- 문제 8
  -- mod(customor_id, 50) : 나머지
  -- floor(customor_id/50) : 몫

select * from customer;
select
	customer_id,
	last_name ||', '|| first_name as full_name,
	email,
	mod(customer_id, 50) as mod_50
from
	customer
where 
	mod(customer_id, 50) = 0;
	


-- 문제 10번
select * from city limit 10;

select 
	count(*)
from 
	city c;

select
	count (distinct city)
from
	city;
	

select
	city
from
	city
group by
	city
having
	count(city) > 1;


  -- 중복된 것 찾기
select 
	*
from
	city
where 
	city = 'London'
	

select 
	max(city)
from
	city;
	

select 
	count(distinct city_id)
from
	city;
	

--  문제 10
select * from actor limit 10;

select
	distinct upper(first_name || ' ' || last_name) as full_name
from
	actor;