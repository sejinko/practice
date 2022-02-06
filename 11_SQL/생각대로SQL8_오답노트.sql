-- 巩力 2
select * from payment limit 10;

select
	c.first_name,
	c.last_name
from
	(
		select 
			customer_id,
			sum(amount) as sum_amount,
			row_number() over (order by sum(amount) desc) as rnum
		from
			payment p 
		group by
			customer_id
	) as db
join customer c on c.customer_id = db.customer_id
where 
	rnum = 1;

	

-- 巩力 5
select 
	*
from
(	
	select 
		extract(year from date(payment_date)) as yr,
		extract(month from date(payment_date)) as mon,
		sum(amount) as sum_amount,
		coalesce(lag(sum(amount)) over (order by extract(month from date(payment_date))), 0) as pre_mon_amount,
		sum(amount) - coalesce (lag(sum(amount)) over (order by extract(month from date(payment_date))),0) as gap 
	from
		payment p 
	group by
		extract(year from date(payment_date)),
		extract(month from date(payment_date))
) as db
where db.gap < 0;
	

-- 巩力 7
select * from inventory limit 10;

select 
	i.store_id,
	sum(amount),
	row_number() over(order by sum(amount) desc) as rnumber,
	rank() over(order by sum(amount) desc) as ranknumber,
	dense_rank() over(order by sum(amount) desc) as densenumber
from 
	payment p 
	join rental r on p.rental_id = r.rental_id 
	join inventory i on r.inventory_id =i.inventory_id 
	join store s on i.store_id = s.store_id 
group by 
	i.store_id
	

-- 巩力 9
select 
	*
from 
(
	select
		name,
		title,
		count(distinct rental_id) as cnt,
		row_number() over(partition by name order by count(distinct rental_id) desc, title asc) as rnum
	from 
	(
		select 
			r.rental_id,
			i.film_id,
			f2.title,
			fc.category_id,
			c."name"
		from
			rental r
			join inventory i on i.inventory_id = r.inventory_id 
			join film f2 on f2.film_id = i.film_id 
			join film_category fc  on fc.film_id = i.film_id 
			join category c on c.category_id  = fc.category_id 
	) as db
	group by name, title
) as db
where rnum <=5;


-- 巩力 10



select 
	"name,",
	sum(amount) as sum_amount,
	first_value (c."name")
	