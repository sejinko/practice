-- ��� ���� ������ ���϶�
-- ��� �����ϴ� ������ �����ϰ��� �ϴ� ����

select 
	film_id,
	title
from
	film
except
	select
		distinct inventory.film_id,
		title
	from
		inventory
	inner join
		film
	on film.film_id = inventory.film_id 
	order by title;
	


-- ���� 1
select 
	film_id,
	title
from 
	film a
where not exists
(
	select
		1
	from
		inventory b,
		film c
	where b.film_id = c.film_id
	and	a.film_id = c.film_id 
);


	
-- �� ���� ����
select 
	film_id,
	title
from 
	film a
where not exists
(
	select
		1
	from
		inventory b
	where 1=1
	and	a.film_id = b.film_id 
);


	