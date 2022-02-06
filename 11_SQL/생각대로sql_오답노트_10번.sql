
from
(
	select 
		name,
		sum(amount) as sum_amount,
		first_value(c.name) over(order by sum(amount) asc rows between preceding and unbounded following) as first_values,
		last_value(c.name) over(order by sum(amount) desc rows between preceding and unbounded following) as last_values
	from
		payment p 
		join rental r on p.rental_id = r.rental_id 
		join inventory i on r.inventory_id = i.inventory_id 
		join film_category fc on i.film_id = fc.film_id 
		join category c on c.category_id = fc.category_id 
	group by c."name" 
) as db


select 
*
from 
(
	select 
		name,
		sum(p.amount) as sum_amount,
		first_value(c.name) over(order by sum(p.amount) asc rows between unbounded preceding and unbounded following) as first_values,
		last_value(c.name) over(order by sum(p.amount) asc rows between unbounded preceding and unbounded following) as last_values
	from
		payment p 
		join rental r on p.rental_id = r.rental_id 
		join inventory i on r.inventory_id = i.inventory_id 
		join film_category fc on i.film_id = fc.film_id 
		join category c on c.category_id = fc.category_id 
	group by c."name"
	order by sum(p.amount) desc
	limit 5
) as db
where
 db.name in (first_values, last_values);
