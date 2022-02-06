-- ���� 1��
select 
	*
from 
	rental;
	
-- ���� 2��
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
	
-- ���� 3��
select 
	customer_id,
	first_name,
	last_name
from 
	customer
where 
	customer_id = 2;
	

-- ���� 4��
select * from payment;

select 
	payment_id,
	amount
from
	payment
where 
	payment_id = 17510;
	

-- ���� 5��
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

-- ���� 6�� -- ��ƴ ǰ
select 
	rating
from
	film
group by
	rating;

select 
	count(rating) over(partition by rating) as num -- ?? �̰� �ƴѰ� ������? 
from 
	film;


-- ���� 7�� -- ��ǰ
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


-- ���� 8��
select * from customer;
select
	customer_id * 50,
	first_name,
	last_name,
	email
from 
	customer;

	
-- ���� 9��
select * from film;
select 
	title
from
	film
where 
	length(title) = 8;

-- ���� 10��
select * from city;
select 
	distinct count(city)
from
	city;


-- ���� 11��
select * from actor;
select 
	distinct upper(concat(concat(last_name, ' '), first_name))
from
	actor;

-- ���� 12��
  -- ���� 1
select * from customer;
select
	count(active)
from 
	customer
where 
	active = 0;
  -- ���� 2
    -- �̷��� �ص� �Ǵ°�?
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


-- ���� 13��
select * from customer;
  -- ���� 1
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
  -- ���� 2
select 
	count(store_id)
from
	customer
where 
	store_id=1;


-- ���� 14��
select * from rental;
select 
	count(to_char(return_date, 'yyyymmdd'))
from
	rental
where 
	to_char(return_date, 'yyyymmdd') = '20050620'
	

-- ���� 15��
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


-- ���� 16��
select * from language;


-- ���� 17��
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

-- ���� 18��
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


-- ���� 19
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

-- ���� 20��
select * from actor;
select
	first_name as firstname,
	last_name as lastname
from
	actor;
	
	

	
	
