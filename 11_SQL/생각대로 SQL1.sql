-- 문제 1번
select 
	*
from 
	rental;
	
-- 문제 2번
select
	film_id,
	title,
	rental_duration,
	length
from 
	film
where 
	rental_duration >= 4
	and length >= 120;
	
-- 문제 3번
select 
	customer_id,
	first_name,
	last_name
from 
	customer
where 
	customer_id = 2;
	

-- 문제 4번
select * from payment;

select 
	payment_id,
	amount
from
	payment
where 
	payment_id = 17510;
	

-- 문제 5번
select * from film;
select * from film_category;
select * from category;

select 
	name,
	category_id
from 
	category
where 
	name = 'Sci-Fi';

-- 문제 6번 -- 반틈 품
select 
	rating
from
	film
group by
	rating;

select 
	count(rating) over(partition by rating) as num -- ?? 이거 아닌거 같은데? 
from 
	film;


-- 문제 7번 -- 못품
select * from rental;

select 
	*
from
	rental
(
	select 
		return_date - rental_date
	from
		rental
	where 
		(return_date - rental_date) >= 10
);
	

select 
	to_char(return_date - rental_date, 'dd')
from
	rental
where 
	to_char(return_date - rental_date, 'dd') > 03;


-- 문제 8번
select * from customer;
select
	customer_id * 50,
	first_name,
	last_name,
	email
from 
	customer;

	
-- 문제 9번
select * from film;
select 
	title
from
	film
where 
	length(title) = 8;

-- 문제 10번
select * from city;
select 
	distinct count(city)
from
	city;


-- 문제 11번
select * from actor;
select 
	distinct upper(concat(concat(last_name, ' '), first_name))
from
	actor;

-- 문제 12번
  -- 정답 1
select * from customer;
select
	count(active)
from 
	customer
where 
	active = 0;
  -- 정답 2
    -- 이렇게 해도 되는가?
select 
	count(a.*)
from
(
	select
		*
	from 
		customer
	where 
		active = 0
) a;


-- 문제 13번
select * from customer;
  -- 정답 1
select 
	count(a.*)
from
(
	select
		*
	from
		customer
	where 
		store_id = 1
) a;
  -- 정답 2
select 
	count(store_id)
from
	customer
where 
	store_id=1;


-- 문제 14번
select * from rental;
select 
	count(to_char(return_date, 'yyyymmdd'))
from
	rental
where 
	to_char(return_date, 'yyyymmdd') = '20050620'
	

-- 문제 15번
select * from film;
select 
	a.film_id,
	a.release_year,
	a.rating,
	a.rental_duration
from
	film a
where 
	a.release_year = 2006
and a.rating = 'G'
and a.rental_duration = 3;

select 
	*
from
	film a
where 
	a.release_year = 2006
and a.rating = 'G'
and a.rental_duration = 3;


-- 문제 16번
select * from language;


-- 문제 17번
select * from film;
select 
	film_id,
	title,
	description,
	rental_duration
from
	film
where 
	rental_duration >= 7
order by 
	rental_duration desc;

-- 문제 18번
select
	film_id,
	title,
	description,
	rental_duration
from
	film
where 
	rental_duration = 3
or	rental_duration = 5
order by
	rental_duration desc;


-- 문제 19
select * from actor limit 10;
select 
	actor_id,
	last_name,
	first_name
from
	actor
where 
	first_name = 'Nick'
or	last_name = 'Hunt';

-- 문제 20번
select * from actor;
select
	first_name as firstname,
	last_name as lastname
from
	actor;
	
	

	
	
