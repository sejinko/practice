-- 문제 1번
select * from film limit 10;
select 
	film_id,
	case when length >= 180 then 'over_length180' as f rating = 'R' then 
	as flag,
	rating = 'R' as flag
from 
	film f 
where 
	length >= 180
or rating = 'R'


-- 문제 1번 복귀
select
	film_id,
	'over_length180' as flag
from
	film_actor;


select
	film_id,
	'over_length180' as flag
from
	film_actor
where
	film_id in
	(
		select
			film_id 
		from
			film f2 
		where 
			length >=180)
union
select 
	film_id,
	'rating_R' as flag
from
	film f 
where 
	film_id in(
	select 
		film_id 
	from
		film
	where 
		rating = 'R'
	)
	

-- 문제 4번
select * from film limit 10;
select * from film_category limit 10;
select * from category limit 10;

select 
	film_id
from
	film f
except
select 
	f.film_id
from
	film f 
	inner join film_category fc on f.film_id = fc.film_id 
	inner join category c on fc.category_id = c.category_id 
where 
	c.name in ('Action', 'Animation', 'Horror');


-- 8번 문제
select * from customer limit 10;
select * from payment limit 10;
select * from country limit 10;
select * from address limit 10;
select * from city limit 10;
select * from rental limit 10;


select 
	c2.country,
	c.city,
	sum(amount)
from
	address a
	inner join city c on a.address_id = c.city_id
	inner join country c2 on c.country_id = c2.country_id 
	inner join customer c3 on a.address_id = c3.address_id 
	inner join payment p on p.customer_id = c3.customer_id 
group by 
	c2.country, c.city
union all
select 
	c2.country,
	null,
	sum(amount)
from
	address a
	inner join city c on a.address_id = c.city_id
	inner join country c2 on c.country_id = c2.country_id 
	inner join customer c3 on a.address_id = c3.address_id 
	inner join payment p on p.customer_id = c3.customer_id 
group by 
	c2.country
union all
select 
	null,
	null,
	sum(amount)
from
	address a
	inner join city c on a.address_id = c.city_id
	inner join country c2 on c.country_id = c2.country_id 
	inner join customer c3 on a.address_id = c3.address_id 
	inner join payment p on p.customer_id = c3.customer_id 
order by
	country desc;
	
	