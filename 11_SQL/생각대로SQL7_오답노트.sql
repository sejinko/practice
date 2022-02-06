-- 문제 3번
select * from payment limit 10;

select 
	c3.country,
	c2.city,
	sum(p.amount)
from
	payment p 
	join customer c on p.customer_id = c.customer_id 
	join address a on c.address_id = a.address_id 
	join city c2 on c2.city_id = a.city_id 
	join country c3 on c3.country_id = c2.country_id 
group by 
grouping sets 
((c3.country, c2.city),(c3.country),())
order by 
	country desc, city desc;


-- 문제 4
select 
	country,
	city,
	sum(amount)
from
	payment p 
	join customer c on p.customer_id = c.customer_id 
	join address a on c.address_id = a.address_id 
	join city c2 on a.city_id = c2.city_id 
	join country c3 on c2.country_id = c3.country_id
group by
rollup
	(country, city)
order by 
	country desc, city desc;
	

-- 문제 5
select * from actor limit 10;

select 
	actor_id,
	count(distinct film_id) as cnt
from 
	film_actor fa
group by
rollup 
	(actor_id);
	

-- 문제 7
select 
	language_id,
	release_year,
	count(distinct film_id) as cnt
from
	film f 
group by
grouping sets
	((language_id,release_year),(release_year),())
	

-- 문제 9
select * from store limit 10;
select * from customer limit 10;
	
select 
	store_id,
	active,
	count(distinct customer_id) as cnt
from
	customer c
group by
grouping sets
	((store_id,active),(active))
	