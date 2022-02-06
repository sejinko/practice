-- 巩力2
select * from customer limit 10;

select 
	c.customer_id,
	c.first_name,
	c.last_name,
	c.email,
	a.address,
	a.district,
	a.postal_code,
	a.phone,
	ct.city
from
	customer c
	join address a on c.address_id = a.address_id 
	join city ct on a.city_id = ct.city_id


-- 巩力3
select * from city limit 10;
select * from customer limit 10;
select * from address limit 10;

select 
	c.first_name,
	c.last_name,
	c.email,
	a.phone,
	ct.city
from
	customer c
	inner join address a on c.address_id = a.address_id
	inner join city ct on a.city_id = ct.city_id 
where 
	ct.city in ('Lima');


-- 巩力 10
select * from customer limit 10;
select * from country limit 10;
select * from city limit 10;

select 
	c.first_name || ' ' || last_name,
	c.email,
	a.phone,
	ct2.country,
	ct.city
from
	customer c
	inner join address a on c.address_id = a.address_id 
	inner join city ct on a.city_id = ct.city_id
	inner join country ct2 on ct.country_id = ct2.country_id 
where 
--	ct2.country != 'china';
	ct2.country not in ('china');

-- 巩力 13
select * from actor limit 10;
select * from film limit 10;
select * from film_actor limit 10;

select 
	a.actor_id,
	concat(concat(a.first_name, ' '), a.last_name) as full_name,
	angels_actor.title,
	case when angels_actor.title = 'Angels life' then 'Y'
		 else 'N'
	end as angelslife_flag
from
	actor a 
left outer join
(select
	f.film_id,
	f.title,
	fa.actor_id
from
	film f
	join film_actor fa on f.film_id = fa.film_id 
where title = 'Angels Life') as angels_actor on a.actor_id = angels_actor.actor_id;


-- 巩力 14
select * from rental limit 10;
select * from staff limit 10;

select 
	*,
	c.first_name || ' ' || c.last_name as customer_fullname,
	s.first_name || ' ' || c.last_name as staff_fullname
from	
	rental r
	inner join staff s on r.staff_id = s.staff_id 
	inner join customer c on c.customer_id = c.customer_id
where 
	date(r.rental_date) between '2005-06-01' and '2005-06-14'
and (c.first_name || ' ' || c.last_name = 'Gloria Cook' or s.first_name || ' ' || s.last_name = 'Mike Hillyer');
	



	