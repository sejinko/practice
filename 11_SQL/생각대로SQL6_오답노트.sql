-- ���� 1��
select * from customer limit 10;

select 
	first_name,
	last_name
from
	customer c
where 
	customer_id in
	(
		select
			customer_id
			--, sum(amount) as sum_amount
		from
			payment p 
		group by
			customer_id
		order by 
			sum(amount) desc 
		limit 1
	)
	
	
-- ���� 2��
select * from rental limit 10;
select * from inventory limit 10;	
	
select 
	category_id,
	name
from
	category c 
where exists
	(
		select
			1
		from
			rental r 
			join inventory i2 on r.inventory_id = i2.inventory_id 
			join film_category fc on i2.film_id = fc.film_id 
		where c.category_id = fc.category_id
	)
	
-- ���� 2�� �ٽ� Ǯ���
	
select 
	category_id,
	name
from
	category c 
where exists
	(
		select 
			1
		from
			rental r 
			join inventory i2 on r.inventory_id = i2.inventory_id 
			join film_category fc on i2.film_id = fc.film_id 
		where 
			c.category_id = fc.category_id 
	)		
			

-- ���� 3��
select * from rental limit 10;
select * from inventory limit 10;	
select * from film_category limit 10;

select 
	category_id,
	name
from
	category c 
where category_id = any 
	(
		select
			category_id 
		from
			rental r 
			inner join inventory i2 on r.inventory_id = i2.inventory_id 
			inner join film_category fc on i2.film_id = fc.film_id
	)

	
-- ���� 3�� ����
select 
	category_id,
	name
from
	category c
where category_id > any 
	(
		select 
			category_id
		from
			rental r2 
			join inventory i on r2.inventory_id = i.inventory_id 
			join film_category fc on fc.film_id = i.film_id 
		where 
			fc.category_id in (1,2)
	)
	
-- ���� 5��
select * from rental limit 10;
select * from customer limit 10;	

select 
	first_name,
	last_name
from
	customer c 
where
	customer_id in
	(
		select 
			customer_id
		from
			rental
		group by
			customer_id
		order by 
			count(rental_id) desc
		limit 1
	);
			
			
		
-- ���� 6��
select 
	*
from 
	film f
where
	film_id 
not in
	(
		select 
			film_id
		from
			film_category fc
		 
	)
	
select
	*
from 
	film f 
where 
	not exists
	(
		select 
			1
		from
			film_category fc
		where 
			fc.film_id = f.film_id 
	)
	
	
-- in �� exists�� ������ ����
-- not in �� not exists�� ���̰� �ִٰ� ���� ��
-- not exists �� not in + null �̶�� �����ϸ� ��

select 	*
from 
	address a 
where 
	address2 = ''
order by 
	address2 desc;

select 	*
from 
	address a 
where 
	address2 = ''
order by 
	address2 asc;

	
select 
	*
from
	address a
where 
	a.address2 in 
	(
		select 
			null);
			

select 
	*
from 
	address a
where 
	a.address2 not in
	(
		select ''
	);
	
			
select 
	*
from 
	address a
where 
	not exists
	(
		select 
			1
		from
			(
				select
					'' as a
			) as db
		where
			db.a = a.address2 
	);
			
	
			

			
	