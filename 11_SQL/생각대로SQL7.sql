-- 문제 1번
select * from inventory limit 10;

select 
	store_id,
	inventory_id,
	sum(inventory_id)
from
	inventory
group by
grouping sets 
	((store_id, inventory_id),(store_id), ())
order by 
	store_id desc, inventory_id desc, sum(inventory_id) desc;


-- 문제 2번
select * from inventory limit 10;

select 
	store_id,
	inventory_id,
	sum(inventory_id)
from
	inventory i 
group by
rollup 
(store_id, inventory_id)
order by 
	store_id desc, inventory_id desc;


-- 문제 3번
select * from payment limit 10;
select * from country limit 10;
select * from city limit 10;

select 
	country,
	city,
	sum(amount)
from
	country c 
	city
group by
grouping sets 
	((payment, country, city), (payment, country), (payment), ())
	

-- 문제 4번

group by 
rollup 
	(payment, country, city)

	
-- 문제 5번
select * from actor limit 10;
select * from film_actor limit 10;
select * from film limit 10;

select 
	a.first_name || ' ' || a.last_name as full_name,
	sum(f.film_id)
from
	film f 
	inner join film_actor fa on f.film_id = fa.film_id 
	inner join actor a on fa.actor_id = a.actor_id 
group by
rollup
	(a.first_name || ' ' || a.last_name);
	

-- 문제 6번
select * from customer limit 10;
select * from address limit 10;

select 
	country,
	city,
	sum(customer_id)
from
	~~~
group by
grouping sets 
((country, city customer_id), (country, city), (country), ())


-- 문제 7번
select * from film limit 10;
select * from language limit 10;

select 
	language_id,
	language.name
	release_id,
	sum(film_id),
	sum(release_id)
from
	film f 
group by
rollup 
	(language_id, release_id, film_id)


-- 문제 8번
select * from payment limit 10;

select 
	to_char(payment_date, 'yyyy'),
	to_char(payment_date, 'dd'),
	sum(payment_id)
from
	payment p 
group by
grouping sets 
((to_char(payment_date, 'yyyy'), to_char(payment_date, 'dd')), (to_char(payment_date, 'yyyy')), ());
 
 
-- 문제 9번
select * from customer limit 10;
select * from store limit 10;

select
	c.store_id,
	active,
	sum(customer_id)
from
	customer c
	inner join store s on c.store_id = s.store_id
group by 
grouping sets((c.store_id, active, customer_id), (c.store_id, active), (c.store_id), ());


-- 문제 10번
-- 문제 9번
select * from customer limit 10;
select * from store limit 10;

select
	c.store_id,
	active,
	sum(customer_id)
from
	customer c
	inner join store s on c.store_id = s.store_id
group by 
rollup 
	(c.store_id, active, customer_id)
