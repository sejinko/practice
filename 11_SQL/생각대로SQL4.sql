-- 巩力1
select * from staff limit 10;
select * from store limit 10;

select
	s.store_id,
	count(staff_id) as staff_count
from 
	staff s
	join store st on s.staff_id = s.staff_id
group by
	s.store_id;
	

select
	*,
	count(s.staff_id) over()
from 
	staff s;
	


-- 巩力2
select * from film limit 10;

select 
	f.rating,
	count(f.rating)
from 
	film f 
group by
	f.rating;
	

-- 巩力 3
select * from actor limit 10;
select * from film limit 10;
select * from film_actor limit 10;

select 
	f.film_id,
	f.title,
	count(fa.actor_id)
from
	film f 
	join film_actor fa on f.film_id = fa.film_id 
group by
	f.film_id
having
	count(fa.actor_id) > 10;


-- 巩力 4
select * from actor limit 10;
select * from film limit 10;
select * from film_actor limit 10;

select 
	a.actor_id,
	a.first_name || ' ' || a.last_name as fullname,
	count(f.film_id)
from
	actor a
	join film_actor f on a.actor_id = f.film_id 
group by
	a.actor_id;


-- 巩力 5
select * from country limit 10;
select * from customer limit 10;

select 
	ctry.country,
	count(c.customer_id)
from
	country ctry
	join customer c on ctry.country_id = c.customer_id
group by 
	ctry.country
order by count(c.customer_id) desc;


-- 巩力 6
select * from inventory limit 10;
select * from store limit 10;
select * from film limit 10;

select 
	f.film_id,
	f.title,
	count(i.inventory_id)
from
	inventory i
	inner join film f on i.film_id = f.film_id 
group by 
	f.film_id
having
	count(i.inventory_id) >= 3;
	

-- 巩力 7
select * from rental limit 10;
select * from customer limit 10;


select 
	c.customer_id,
	c.first_name || ' ' || c.last_name as full_name,
	count(rental_id)
from
	rental r 
	inner join customer c on r.customer_id = c.customer_id 
group by
	c.customer_id
order by 
	count(rental_id) desc
limit 1;


-- 巩力 8
select * from rental limit 10;

select
	customer_id,
	count(date(r.rental_date) = '2005-05-26')
from
	rental r 
group by 
	customer_id
having
	count(date(r.rental_date) = '2005-05-26') >= 2
order by 
	count asc;

-- 巩力 9
select * from film_actor limit 10;


select 
	actor_id,
	count(film_id)
from
	film_actor
group by
	actor_id
order by
	count(film_id) desc
limit 5;


-- 巩力 10
select * from payment limit 10;

select 
	customer_id,
	count(payment_id),
	sum(amount)
from
	payment p 
group by
	customer_id 
having
	count(payment_id) >= 2
and	sum(amount) >= 10
order by 
	sum(amount) desc;


-- 巩力 11

	

