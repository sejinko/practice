-- 문제1, 2
select * from customer limit 10;
select * from address limit 10;
select * from city limit 10;
select
	a.address_id,
	a.customer_id,
	a.first_name,
	a.last_name,
	a.email,
	b.address,
	b.district,
	b.postal_code,
	b.phone,
	c.city
from
	customer a,
	address b,
	city c
where
 	a.address_id = b.address_id
and b.city_id = c.city_id;


-- 문제3
select * from city limit 10 ;
select * from customer limit 10;
select * from address limit 10 ;
select
	a.address_id,
	a.first_name,
	a.last_name,
	a.email,
	b.phone
from
	customer a,
	address b
where
	a.address_id = b.address_id;


-- 문제4
select * from rental limit 10;
select * from customer limit 10;
select * from employee limit 10;

select 
 	a.*,
 	b.first_name || ' ' || b.last_name as 고객이름,
 	c.first_name || ' ' || c.last_name as 직원이름
from
 	rental a,
 	customer b,
 	employee c
where 
 	a.customer_id = b.customer_id
and a.staff_id = c.manager_id;


-- 문제5
select * from customer limit 10;
select * from address limit 10;
select * from city limit 10;

select 
	a.address_id,	
	b.address,
	b.address2,
	b.postal_code,
	b.phone,
	c.city
from
	customer a,
	address b,
	city c
where
	a.address_id = b.address_id 
and	b.city_id = c.city_id
and a.email = 'seth.hannon@sakilacustomer.org';


-- 문제 6
select * from payment limit 10;
select * from rental limit 10;
select * from customer limit 10;
select * from employee limit 10;

select
	a.payment_id,
	c.first_name || ' ' || c.last_name as customer_name,
	c.first_name || ' ' || c.last_name as staff_name,
	b.rental_id,
	a.amount
from 
	payment a,
	rental b,
	customer c,
	employee d
where 
	a.customer_id = b.customer_id 
and	a.staff_id = b.staff_id 
and a.rental_id = b.rental_id;


-- 문제 7  -- 못품
select * from film limit 10;
select * from film_actor limit 10;
select * from film_category limit 10;
select * from actor limit 10;

select
	*
from
	film_actor;
with tmp1 as 
select 
	a.film_id,
	b.actor_id,
	a.title,
	a.release_year,
	a.rental_rate,
	a.length
from
	film a,
	film_actor b,
	film_category c,
	actor d
where 
	b.actor_id = d.actor_id 
and	a.film_id = b.film_id;


-- 문제 8
select * from address limit 10;
select * from city limit 10;

select 
	a.address,
	a.address2,
	a.district,
	b.city
from
	address a, 
	city b
where 
	a.city_id = b.city_id 
	
	
-- 문제 9
select * from customer limit 10;
select * from address limit 10;
select * from city limit 10;
select * from country limit 10;

select
	a.customer_id,
	a.first_name,
	a.last_name,
	a.email,
	b.address,
	b.district,
	b.phone,
	c.city,
	d.country
from 
	customer a,
	address b,
	city c,
	country d
where
	a.address_id = b.address_id 
and	b.city_id = c.city_id ;
	
