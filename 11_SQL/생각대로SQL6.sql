-- 문제1
select * from payment limit 10;
select * from customer limit 10;
select * from 

select 
	c.customer_id,
	c.first_name || ' ' || c.last_name as fullname,
	sum(p.amount)
from
	payment p
	left outer join customer c on p.customer_id = c.customer_id 
group by
	c.customer_id
order by 
	sum desc
limit 1;
	
  -- 정답
 select 
 	c.customer_id,
 	c.first_name || ' ' || c.last_name as fullname,
 	(
 		select
 			p.customer_id,
 			sum(p.amount)
 		from
 			payment p 
 		group by
 			p.customer_id
 	)
from 
	customer c;
 			
 			
-- 문제2
select * from category limit 10;
select * from rental limit 10;
select * from customer limit 10;
select * from film_category limit 10;
select * from film limit 10;

select 
	customer_id,
	count(rental_id)
from
	rental r
	left outer join payment p on r.rental_id = p.rental_id 
	left outer join customer c on r.customer_id = c.customer_id 
	left outer join film_actor fa on r.
group by
	customer_id
having 
	count(rental_id) >= 1
order by 
	count ;





-- 문제 5
select * from rental limit 10;

select 
	c.customer_id,
	c.first_name || ' ' || last_name as full_name,
	sum(r.rental_id) as rental_sum
from
	rental r 
	left outer join customer c on r.customer_id = c.customer_id 
group by 
	c.customer_id
order by 
	rental_sum desc
limit 1;


-- 문제 6
select * from film limit 10;
select * from film_category limit 10;
select * from category limit 10;

select
	f.title,
	c.name
from
	film f
	inner join film_category fa on f.film_id = fa.film_id 
	inner join category c on fa.category_id = c.category_id
where 
	name = '';
	
	
	
	




